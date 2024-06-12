import torch
import torch.nn as nn
import torch.nn.functional as F
from opt_einsum import contract as einsum
from einops import rearrange, repeat
import numpy as np
import math

try:
    from flash_attn import flash_attn_qkvpacked_func, flash_attn_func
except:
    flash_attn = None

def broadcast_tokens_to_atoms(tokens, transition_tensor):
    """
        tokens              : shape (b x i x d),
        tensition_tensor    : shape (b x j x i), 

        -----------------------------------------
        (b x i x d) * (b x i x j)

        returns             : shape (b x j x d)

    """
    return torch.bmm(transition_tensor, tokens)

def broadcast_tokens_to_atoms_2D(tokens, transition_tensor):
    """
        tokens              : shape (b x i x i x d),
        tensition_tensor    : shape (b x j x i), 

        -----------------------------------------
        (b x i x i d) * (b x i x 1 x j) => b x i x i x j

        returns             : shape (b x j x d)

    """
    tokens = torch.matmul(transition_tensor.unsqueeze(1), tokens)
    return torch.matmul(transition_tensor.unsqueeze(1), tokens.permute(0, 3, 1, 2)).permute((0,3,2,1))

def mean_atom_aggregate(atoms, transition_tensor, dim=1):
    """
        atoms              : shape (b x j x d),
        tensition_tensor    : shape (b x j x i), 

        -----------------------------------------
    
        returns             : shape (b x i x d)

    """
    aggregate_atoms = einsum('bjd,bji->bjid', atoms, transition_tensor) # (b x i x 1 x d) * (b x i x j x 1) -> (b x i x j x d)
    return aggregate_atoms.sum(dim) / atoms.shape[1]

def extract_representative_atoms(atoms, transition_tensor, dim=1):
    """
        atoms              : shape (b x j x d),
        tensition_tensor    : shape (b x i x j), 

        -----------------------------------------
    
        returns             : shape (b x i x d)

    """
    aggregate_atoms = einsum('bjd,bij->bid', atoms, transition_tensor) # (b x i x 1 x d) * (b x i x j x 1) -> (b x i x j x d)
    return aggregate_atoms

def extract(a, t, x_shape):
    batch_size = t.shape[0]
    a = a.cpu()
    out = a.gather(-1, t.cpu())
    return (out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))).to(t.device)

class NoiseSchedular:
    def __init__(self, timesteps, device):
        self.timesteps = timesteps
        self.device = device

        self.betas = self.cosine_beta_schedule()
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = torch.nn.functional.pad(self.alphas_cumprod[:-1], (1,0), value=1.0)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)

        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
        self.posterior_varience = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)


    def cosine_beta_schedule(self, s=0.008):
        """
        cosine schedule as proposed in https://arxiv.org/abs/2102.09672
        """
        steps = self.timesteps + 1
        x = torch.linspace(0, self.timesteps, steps, device=self.device)
        alphas_cumprod = torch.cos(((x / self.timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)


class AdaLN(nn.Module):
    def __init__(self, a_dim, s_dim):
        super(AdaLN, self).__init__()
        self.a_norm = nn.LayerNorm(a_dim, elementwise_affine=False, bias=False)
        self.s_norm = nn.LayerNorm(s_dim, bias=False)

        self.proj_bias = nn.Linear(s_dim, a_dim)
        self.proj_nobias = nn.Linear(s_dim, a_dim, bias=False)

    def reset_parameter(self):
        nn.init.zeros_(self.proj_bias.weight)
        nn.init.zeros_(self.proj_bias.bias)
        nn.init.zeros_(self.proj_nobias.weight)        

    def forward(self, a, s):
        a = self.a_norm(a)
        s = self.s_norm(s)
        a = F.sigmoid(
            self.proj_bias(s)* a + self.proj_nobias(s) # may need to transpose for matmul
        )
        return a
    

class TriangleMultiplicationOutgoing(nn.Module): # N x N x C -> N x N x C
    def __init__(self, d_model, channel):
        super(TriangleMultiplicationOutgoing, self).__init__()
        self.norm_layer = nn.LayerNorm(d_model)
        self.proj_layer = nn.Linear(d_model, channel*4, bias=False)
        self.gate_layer = nn.Linear(d_model, d_model, bias=False)
        self.ab_norm_layer = nn.LayerNorm(channel)
        self.ab_layer = nn.Linear(channel, d_model, bias=False)

    def forward(self, x):
        x = self.norm_layer(x)
        a1, a2, b1, b2 = self.proj_layer(x).chunk(4, -1)
        g = self.gate_layer(x)
        a1 = F.sigmoid(a1) * a2
        
        b1 = F.sigmoid(b1) * b2

        out = self.ab_layer(self.ab_norm_layer(einsum('bikd,bjkd->bijd', a1, b1)))
        g = F.sigmoid(g)
        x = g * out
        return x

class TriangleMultiplicationIncoming(nn.Module):
    def __init__(self, d_model, channel):
        super(TriangleMultiplicationIncoming, self).__init__()
        self.norm_layer = nn.LayerNorm(d_model)
        self.proj_layer = nn.Linear(d_model, channel*4, bias=False)
        self.gate_layer = nn.Linear(d_model, d_model, bias=False)
        self.ab_norm_layer = nn.LayerNorm(channel)
        self.ab_layer = nn.Linear(channel, d_model, bias=False)

    def forward(self, x): 
        x = self.norm_layer(x)
        a1, a2, b1, b2 = self.proj_layer(x).chunk(4, -1)
        g = self.gate_layer(x)
        a1 = F.sigmoid(a1) * a2
        b1 = F.sigmoid(b1) * b2
        out = self.ab_layer(self.ab_norm_layer(einsum('bkid,bkjd->bijd', a1, b1)))
        g = F.sigmoid(g)
        x = g * out
        return x

class TriangleSelfAttentionStartingNode(nn.Module):
    def __init__(self, d_model, channel, nheads, device=None):
        super(TriangleSelfAttentionStartingNode, self).__init__()

        assert channel % nheads == 0
        self.d_model = d_model
        self.channel = channel
        self.nheads = nheads
        self.proj_norm = nn.LayerNorm(d_model)
        self.proj_layer = nn.Linear(d_model, channel*4, bias=False) # for qkv and bias projections and for heads.
        self.proj_bias = nn.Linear(d_model, nheads, bias=False)
        self.out_layer = nn.Linear(channel, d_model, bias=False)

        self.dc = torch.scalar_tensor(self.channel, device=device)

    def forward(self, x): # X size b x n x n x c
        #project inputs
        x = self.proj_norm(x)
        q,k,v,g = self.proj_layer(x).reshape((x.shape[0], self.nheads, x.shape[1], x.shape[2], -1)).chunk(4, -1)
        b = self.proj_bias(x).permute((0, 3, 1, 2))
        #attentions with bias
        score = einsum('bhijc,bhikc->bhjk', q, k) / torch.sqrt(self.dc) 
        score += b
        score= F.softmax(score, 
            dim=-1  
        ).unsqueeze(-1)
        out = (g*einsum('bhijk, bhlkc->bhijc',score, v)).view((q.shape[0], q.shape[2], q.shape[2], self.channel))
        # out projection
        out = self.out_layer(out)
        return out

class TriangleSelfAttentionEndingNode(nn.Module):
    def __init__(self, d_model, channel, nheads, device=None):
        super(TriangleSelfAttentionEndingNode, self).__init__()

        assert channel % nheads == 0

        self.d_model = d_model
        self.channel = channel
        self.nheads = nheads

        self.proj_norm = nn.LayerNorm(d_model)
        self.proj_layer = nn.Linear(d_model, channel*4, bias=False) # for qkv and bias projections and for heads.
        self.proj_bias = nn.Linear(d_model, nheads, bias=False)

        self.out_layer = nn.Linear(channel, d_model, bias=False)

        self.dc = torch.scalar_tensor(self.channel, device=device)

    def forward(self, x): # X size b x n x n x c
        #project inputs
        x = self.proj_norm(x)
        q,k,v,g = self.proj_layer(x).reshape((x.shape[0], self.nheads, x.shape[1], x.shape[2], -1)).chunk(4, -1)
        b = self.proj_bias(x).permute((0, 3, 1, 2))

        #attentions with bias
        score = einsum('bhijc,bhikc->bhjk', q, k) / torch.sqrt(self.dc) 
        score += b.transpose(3, 2)
        score= F.softmax(score, 
            dim=-1  
        ).unsqueeze(-1)
        out = (g*einsum('bhijk, bhlkc->bhijc',score, v)).view((q.shape[0], q.shape[2], q.shape[2], self.channel))
        
        # out projection
        
        out = self.out_layer(out)
        return out

class Transition(nn.Module): # to my understanding just a FF stack?
    def __init__(self, channel, n=4):
        super(Transition, self).__init__()
        self.inp_norm = nn.LayerNorm(channel)
        self.proj_inp = nn.Linear(channel, channel*n*2, bias=False)
        self.proj_out = nn.Linear(channel*n, channel)
        self.swish = nn.SiLU()

    def forward(self, x): # x = b x n x n x c
        x = self.inp_norm(x)
        a, b = self.proj_inp(x).chunk(2, -1)
        x = self.proj_out(
                self.swish(a)* b)
        return x
    
class AttentionPairBias(nn.Module): # the conditional and MPNN crosstalk step
    def __init__(self, ca, cz, cs, nheads=16,device=None):
        super(AttentionPairBias, self).__init__()


        assert ca % nheads == 0
        self.nheads = nheads
        c = int(ca/nheads)

        self.adaLN = AdaLN(ca, cs)
        self.alt_norm = nn.LayerNorm(ca)
        self.pair_norm = nn.LayerNorm(cz)
        self.query_proj = nn.Linear(ca, ca)
        self.kvg_proj = nn.Linear(ca, ca*3, bias=False)
        self.bias_proj = nn.Linear(cz, nheads)
        self.attn_proj = nn.Linear(ca, ca, bias=False)
        self.out_proj = nn.Linear(cs, ca)
        
        self.dc = torch.scalar_tensor(c, device=device)

    def reset_parameters(self):
        nn.init.constant_(self.out_proj.bias, -2.0)

    def forward(self, a, z, beta=0, s=None):# a : b,n,c    z: b,n,n,c 
        batch_size = a.shape[0]
        seq_size = a.shape[1]
        channel_size = a.shape[2]

        if s is not None:
            a = self.adaLN(a, s)
        else:
            a = self.alt_norm(a)

        q = self.query_proj(a).reshape((batch_size, self.nheads, seq_size, -1))
        k, v, g = self.kvg_proj(a).reshape((batch_size, self.nheads, seq_size, -1)).chunk(3, -1)

        b = (self.bias_proj(self.pair_norm(z)) + beta).reshape((batch_size, self.nheads, seq_size, seq_size))
        g = F.sigmoid(g)

        # Attention time

        A = F.softmax(
            (einsum('bhic,bhjc->bhji',q, k)/self.dc) + b , dim=-1
        )

        a = self.attn_proj(
            (g * einsum('bhij,bhic->bhjc',A, v)).view((batch_size, seq_size, -1))
        )

        # conditioning projection from AdaLN-Zero with -2.0 bias inital?
        if s is not None:
            a = F.sigmoid(self.out_proj(s)) * a
        return a


class PairformerLayer(nn.Module):
    def __init__(self, params):
        super(PairformerLayer, self).__init__()
        
        ca = params['ca']
        cz = params['cz']
        cs = params['cs']
        self.device = params['device']
        triangle_channels = params['Triangle_multi_c']
        tri_attention_channel = params['Triangle_attention_channel']
        transition_scale = params['Pairformer_transition_scale']
        drop_p = params['Pairformer_drop_p']
        triangle_nheads = params['Triangle_attention_nheads']
        pair_heads = params['Pairformer_pair_nheads']

        self.TMO_dropout = torch.nn.Dropout(drop_p)
        self.TMI_dropout = torch.nn.Dropout(drop_p)
        self.TAS_dropout = torch.nn.Dropout(drop_p)
        self.TAE_dropout = torch.nn.Dropout(drop_p)

        self.TMO = TriangleMultiplicationOutgoing(cz, triangle_channels)
        self.TMI = TriangleMultiplicationIncoming(cz, triangle_channels)
        
        self.TAS = TriangleSelfAttentionStartingNode(cz, tri_attention_channel, triangle_nheads, self.device)
        self.TAE = TriangleSelfAttentionEndingNode(cz, tri_attention_channel, triangle_nheads, self.device)

        self.transition_z = Transition(cz, transition_scale)
        self.transition_s = Transition(cs, transition_scale)

        self.attention = AttentionPairBias(ca, cz, cs, pair_heads)

    def forward(self, s, z):
        z += self.TMO_dropout(self.TMO(z))
        z += self.TMI_dropout(self.TMI(z))
        z += self.TAS_dropout(self.TAS(z))
        z += self.TAE_dropout(self.TAE(z))
        z += self.transition_z(z)

        s += self.attention(s, z)
        s += self.transition_s(s)
        return s, z

class Pairformer(nn.Module):
    def __init__(self, params):
        super(Pairformer, self).__init__()
        
        self.layers = nn.ModuleList([
            PairformerLayer(params) for i in range(params['Pairformer_nblocks'])
        ])
    
    def forward(self, s, z):
        for layer in self.layers:
            s, z = layer(s, z)
        return s, z
    
class CentreRandomAugmentation(nn.Module):
    """ Algorithm 19 """

    def __init__(self, trans_scale: float = 1.0, device=None):
        super(CentreRandomAugmentation, self).__init__()
        self.device = device

        self.trans_scale = torch.scalar_tensor(trans_scale, device=self.device)

        self.two = torch.scalar_tensor(2, device=self.device)

    def forward(self, coords):
        """
        coords: coordinates to be augmented
        """
        batch_size = int(coords.shape[0])

        # Center the coordinates
        centered_coords = coords - coords.mean(dim=1, keepdim=True)

        # Generate random rotation matrix
        rotation_matrix = self._random_rotation_matrix(batch_size)

        # Generate random translation vector
        translation_vector = self._random_translation_vector(batch_size)
        translation_vector = rearrange(translation_vector, 'b c -> b 1 c')

        # Apply rotation and translation
        augmented_coords = einsum('bni,bji->bnj', centered_coords, rotation_matrix) + translation_vector

        return augmented_coords

    def _random_rotation_matrix(self, batch_size):
        # Generate random rotation angles
        angles = torch.rand((batch_size, 3), device = self.device) * self.two * torch.pi

        # Compute sine and cosine of angles
        sin_angles = torch.sin(angles)
        cos_angles = torch.cos(angles)

        # Construct rotation matrix
        eye = torch.eye(3, device = self.device)
        rotation_matrix = repeat(eye, 'i j -> b i j', b = batch_size).clone()

        rotation_matrix[:, 0, 0] = cos_angles[:, 0] * cos_angles[:, 1]
        rotation_matrix[:, 0, 1] = cos_angles[:, 0] * sin_angles[:, 1] * sin_angles[:, 2] - sin_angles[:, 0] * cos_angles[:, 2]
        rotation_matrix[:, 0, 2] = cos_angles[:, 0] * sin_angles[:, 1] * cos_angles[:, 2] + sin_angles[:, 0] * sin_angles[:, 2]
        rotation_matrix[:, 1, 0] = sin_angles[:, 0] * cos_angles[:, 1]
        rotation_matrix[:, 1, 1] = sin_angles[:, 0] * sin_angles[:, 1] * sin_angles[:, 2] + cos_angles[:, 0] * cos_angles[:, 2]
        rotation_matrix[:, 1, 2] = sin_angles[:, 0] * sin_angles[:, 1] * cos_angles[:, 2] - cos_angles[:, 0] * sin_angles[:, 2]
        rotation_matrix[:, 2, 0] = -sin_angles[:, 1]
        rotation_matrix[:, 2, 1] = cos_angles[:, 1] * sin_angles[:, 2]
        rotation_matrix[:, 2, 2] = cos_angles[:, 1] * cos_angles[:, 2]

        return rotation_matrix

    def _random_translation_vector(self, batch_size: int):
        # Generate random translation vector
        translation_vector = torch.randn((batch_size, 3), device = self.device) * self.trans_scale
        return translation_vector


class RelativePositionEncoding(nn.Module):
    def __init__(self, params):
        super(RelativePositionEncoding, self).__init__()

        self.device = params['device']

        cz = params['cz']
        self.rmax = params['rmax']
        self.smax = params['smax']

        self.r_bins = self.rmax*2+1
        self.s_bins = self.smax*2+1

        self.zero = torch.scalar_tensor(0, dtype=torch.int32, device=self.device)
        self.two = torch.scalar_tensor(2, dtype=torch.int32, device=self.device)

        self.proj_layer = nn.Linear(self.r_bins*2+self.s_bins+1, cz, bias=False)

        self.rmax = torch.scalar_tensor(params['rmax'], dtype=torch.int32, device=self.device)
        self.smax = torch.scalar_tensor(params['smax'], dtype=torch.int32, device=self.device)

    def forward(self, f):
        #create pair_wise matricies
        b_same_chain = torch.eq(f['asym_id'].unsqueeze(1), f['asym_id'].unsqueeze(2))
        b_same_residue = torch.eq(f['residue_idx'].unsqueeze(1), f['residue_idx'].unsqueeze(2))
        b_same_entity = torch.eq(f['entity_id'].unsqueeze(1), f['entity_id'].unsqueeze(2))

        #IDK tbh
        d_residue = torch.where(b_same_chain == False, torch.clip( #could try *-1 to make a not
            f['residue_idx'].unsqueeze(1) - f['residue_idx'].unsqueeze(2) + self.rmax, self.zero, self.two*self.rmax
        ), self.two*self.rmax)

        a_rel_pos = F.one_hot(d_residue, self.r_bins).float()

        d_tokens = torch.where(torch.logical_and(b_same_residue == False, b_same_chain == False), torch.clip( 
            f['token_idx'].unsqueeze(1) - f['token_idx'].unsqueeze(2) + self.rmax, self.zero, self.two*self.rmax
        ), self.two*self.rmax)
        a_rel_token = F.one_hot(d_tokens, self.r_bins).float()

        d_chain = torch.where(b_same_chain == False, torch.clip(
            f['sym_id'].unsqueeze(1) - f['sym_id'].unsqueeze(2) + self.smax, self.zero, self.two*self.smax
        ), self.two*self.smax)

        a_rel_chain = F.one_hot(d_chain, self.s_bins).float()
        
        p = self.proj_layer(
            torch.concat([
                a_rel_pos, a_rel_token, b_same_entity.unsqueeze(-1).float(), a_rel_chain
            ], dim=-1)
        )

        return p


class FourierEmbedding(nn.Module):
    def __init__(self, nd=256, device=None):
        super(FourierEmbedding, self).__init__()
        self.nd = nd
        #generates random weight and biase before trianing
    
    def forward(self, t, shape, device):
        """
            t : batch_size,
            self.w : nd,
            self.b : nd,

            return size [batchsize, nd]
        """
        self.w = torch.rand((shape[0], shape[1], self.nd), device=device)
        self.b = torch.rand((shape[0], shape[1], self.nd), device=device)

        return torch.cos(2*(torch.pi*(t*self.w + self.b)))

class DiffusionCondition(nn.Module): #NOTE: Algo 21
    def __init__(self, params):
        super(DiffusionCondition, self).__init__()

        self.device = params['device']
        cs = params['cs']
        cz = params['cz']
        nd = params['Fourier_n']
        self.sigmadata = params['sigma_data']

        self.rope = RelativePositionEncoding(params)
        
        self.z_proj = nn.Linear(cz*2, cz, bias=False)
        self.z_norm = nn.LayerNorm(cz*2)
        self.transition_z = Transition(cz, n=2)

        self.s_proj = nn.Linear(cs*2, cs, bias=False)
        self.s_norm = nn.LayerNorm(cs*2)
        self.transition_s = Transition(cs, n=2)

        self.fourier_embed = FourierEmbedding(nd, self.device)
        self.n_proj = nn.Linear(nd, cs) #TODO: ? what is the projected dim????
        self.n_norm = nn.LayerNorm(nd)

    def forward(self, t, f, s_inp, s_trunk, z):
        
        # Pair conditions
        z = torch.concat(
            (z, self.rope(f)), dim=-1
        )
        z = self.z_proj(self.z_norm(z))

        z += self.transition_z(z) #TODO : needs to fix this with the for loop Algo 21

        # Single conditions

        s = torch.concat((s_inp, s_trunk), dim=-1)
        s = self.s_proj(self.s_norm(s))
        n = self.fourier_embed(
            0.25 * math.log(t/self.sigmadata), s.shape, self.device
        )
        s += self.n_proj(self.n_norm(n)) ## ????? (b,256) -> b,256,1 or b,1,256 | s : b,n,cs is n == 256 cs is 384?

        s += self.transition_s(s) #TODO : needs to fix this with the for loop Algo 21

        return s, z

class AtomTransformer(nn.Module): #NOTE: Algo 7
    def __init__(self, params, nheads=None, nblocks=None):
        super(AtomTransformer, self).__init__()
        
        self.device = params['device']
        self.nqueries = params['AT_nqueries']
        self.nkeys = params['AT_nkeys']
        self.subset_centres = torch.tensor(params['S_subset_centres'], device=self.device)
        self.diffusion_transformer = DiffusionTransformer(params, params['catoms'], params['catom_pairs'], params['catoms'], nheads=nheads, nblocks=nblocks)
        # self.mask = torch.ones((self.nqueries*3, int(self.nkeys*1.125)))

        self.zero = torch.scalar_tensor(0, device=self.device)
        self.neg = torch.scalar_tensor(-10**10, device=self.device)

    def generate_beta(self, b_shape): #needs to be optimzsed
        l = torch.arange(0, b_shape[1], 1, device=self.device)
        dis = torch.abs(l.unsqueeze(1)-self.subset_centres.unsqueeze(0)).min(dim=1).values
        l_mask = torch.le(dis, self.nqueries)
        m_mask = torch.le(dis, self.nkeys)
        blm = torch.where(torch.logical_and(l_mask, m_mask), self.zero, self.neg)

        return blm.unsqueeze(-1)

    def forward(self, q, c, p):
        b = self.generate_beta(q.shape)
        q = self.diffusion_transformer(q, c, p, b)
        return q

@torch.jit.script #creates a kernal to make this a single opperation rather than 2 or 3 idk if 1+ counts
def IDS_opperation(d, one):
    return one / (one + torch.square(d.sum(-1).unsqueeze(-1)))

class AtomAttentionEncoder(nn.Module):
    def __init__(self, params, ctokens):
        super(AtomAttentionEncoder, self).__init__()
        
        fdim = params['AAE_f_sum_dim']
        catoms = params['catoms']
        catom_pairs = params['catom_pairs']
        cs = params['cs']
        cz = params['cz']
        self.device = params['device']

        self.c_proj = nn.Linear(fdim, catoms, bias=False)
        
        self.P_proj = nn.Linear(3, catom_pairs, bias=False)
        self.IDS_proj = nn.Linear(1, catom_pairs, bias=False)
        self.IDS_mask_proj = nn.Linear(1, catom_pairs, bias=False)

        self.s_proj = nn.Linear(cs, catoms, bias=False)
        self.s_norm = nn.LayerNorm(cs)

        self.z_proj = nn.Linear(cz, catom_pairs, bias=False)
        self.z_norm = nn.LayerNorm(cz)

        self.r_proj = nn.Linear(3, cz)

        self.cp_proj = nn.Linear(catoms, catom_pairs, bias=False)

        self.MLP = nn.Sequential( # TODO: Was this meant to be high dimmension squick MLP or just a flat one?
            nn.ReLU(),
            nn.Linear(catom_pairs, catom_pairs, bias=False),
            nn.ReLU(),
            nn.Linear(catom_pairs, catom_pairs, bias=False),
            nn.ReLU(),
            nn.Linear(catom_pairs, catom_pairs, bias=False),
        )

        self.atom_transformer = AtomTransformer(params, params['AAE_nheads'], params['AAE_nblocks'])

        self.q_proj = nn.Linear(catoms, ctokens, bias=False)

        self.one = torch.scalar_tensor(1, device=self.device)



    def forward(self, f, s_trunk, z_trunk, r=None):
        # Create the atom single conditioning: embed per-atom meta data
        c = self.c_proj(torch.concat(
            [f['ref_pos'], f['ref_charge'].unsqueeze(-1), f['ref_mask'].unsqueeze(-1), f['ref_element'], f['ref_atom_name_chars'].reshape((f['ref_atom_name_chars'].shape[0], f['ref_atom_name_chars'].shape[1], -1))]
            , dim=-1)) # f : {ref_pos : 0, ref_charge : 1, ref_mask : 2, ref_element : 3, ref_atom_name_chars : 4, ref_space_uid : 5}

        #embed offsets between atom reference positions
        d = f['ref_pos'].unsqueeze(1) - f['ref_pos'].unsqueeze(2)
        v = (f['ref_space_uid'].unsqueeze(1) == f['ref_space_uid'].unsqueeze(2)).unsqueeze(-1).float() # Symetry mask
        P = self.P_proj(d) * v

        # Embed pairwise  inverse squared distances, and the valid mask.
        P += self.IDS_proj(
            IDS_opperation(d, self.one)
        ) * v
        P += self.IDS_mask_proj(v) * v

        #Initialise atom single representation as the single conditioning.
        q = torch.clone(c)

        #If provided add trunk embeddings as the single conditioning
        if r is not None: #TODO : REDO THIS!!!!!!
            # Broadcast the single and pair embeddings from trunk
            c += self.s_proj(self.s_norm(
                broadcast_tokens_to_atoms(s_trunk, f['reference_tokens'])
                ))
            P += self.z_proj(
                self.z_norm(
                   broadcast_tokens_to_atoms_2D(z_trunk, f['reference_tokens']
                   )))
            #Add the noisy positions ??? idk why 
            q += self.r_proj(r) ### Might need to redo some of these.
        
        # Add the combined single conditioning to the pair representation
        cp = self.cp_proj(F.relu(c))
        P += (cp.unsqueeze(1) + cp.unsqueeze(2))

        #Run a small MLP on the pair activations
        P += self.MLP(P)

        # Cross Attention Transformer.... Why no AdaLN-zero????
        q = self.atom_transformer(q, c, P)

        # Aggregate per-atom representations to per-token representation.
        a = mean_atom_aggregate(F.relu(self.q_proj(q)), f['reference_tokens'])

        return a, q, c, P


class AtomAttentionDecoder(nn.Module):
    def __init__(self, params, catoms, catom_pairs, ctokens, nblocks=3, nheads=4):
        super(AtomAttentionDecoder, self).__init__()
        self.atom_transformer = AtomTransformer(params, nheads, nblocks)

        self.a_proj = nn.Linear(ctokens, catoms, bias=False)
        self.q_proj = nn.Linear(catoms, 3, bias=False)
        self.q_norm = nn.LayerNorm(catoms)


    def forward(self, a, q_skip, c_skip, p_skip, transition_tensor):
        #broadcast per-token activations to per-atom activations and add the skip connection
        q = self.a_proj(broadcast_tokens_to_atoms(a, transition_tensor)) + q_skip

        # Cross attention transformer
        q = self.atom_transformer(q, c_skip, p_skip)

        #Map to position update
        r = self.q_proj(self.q_norm(q))
        return r

class ConditionedTransitionBlock(nn.Module):
    def __init__(self, ca, cs, n=2):
        super(ConditionedTransitionBlock, self).__init__()

        self.adaLN = AdaLN(ca, cs)
        self.swish = nn.SiLU()
        self.a_proj = nn.Linear(ca, ca*n*2, bias=False)
        self.s_proj_bias = nn.Linear(cs, ca)
        self.b_proj = nn.Linear(ca*n, ca, bias=False)
    
    def reset_parameters(self):
        nn.init.constant_(self.s_proj_bias, -2.0)

    def forward(self, a, s):
        a = self.adaLN(a, s)
        a, a1 = self.a_proj(a).chunk(2, dim=-1)
        b = self.swish(a) * a1

        # output project from adaLN-zero FILM
        a = F.sigmoid(
            self.s_proj_bias(s) * self.b_proj(b)
        )
        return a
    
class DiffusionTransformer(nn.Module):
    def __init__(self, params, ca, cz, cs, nblocks, nheads):
        super(DiffusionTransformer, self).__init__()
        transition_scale = params['DT_transition_scale']
        device = params['device']

        self.nblocks = nblocks
        self.layers = nn.ModuleList([nn.ModuleList([AttentionPairBias(ca, cz, cs, nheads, device), ConditionedTransitionBlock(ca, cs, n=transition_scale)]) for i in range(nblocks)])

    def forward(self, a, s, z, beta=0):
        for i, layer in enumerate(self.layers):
            b = layer[0](a, z, beta, s)
            a = b + layer[1](a, s)
        return a

class DiffusionModule(nn.Module): #NOTE: Algo 20
    def __init__(self, params):
        super(DiffusionModule, self).__init__()
        
        #fdim, catoms, catom_pairs, ctokens, transition_scale=2, nheads=16, nblocks=24
        self.catom_pairs = params['catom_pairs']
        self.ctokens = params['DM_ctokens']
        catoms = params['catoms']
        cs = params['cs']
        self.sigma_data = torch.scalar_tensor(params['sigma_data'], device=params['device'])

        self.diffusion_conditioning = DiffusionCondition(params)
        self.atom_attention_encoder = AtomAttentionEncoder(params, ctokens=params['DM_ctokens'])
        self.diffusion_transformer = DiffusionTransformer(params, self.ctokens, params['cz'], params['cs'], nheads=params['DM_DT_nheads'], nblocks=params['DM_DT_nblocks'])
        self.atom_attention_decoder = AtomAttentionDecoder(params, params['catoms'], params['catom_pairs'], params['DM_ctokens'], params['AAD_nblocks'], params['AAD_nheads'])
        

        self.s_inp_norm = nn.LayerNorm(cs)
        self.s_inp_proj = nn.Linear(cs, self.ctokens, bias=False)

        self.diff_norm = nn.LayerNorm(self.ctokens)

        self.two = torch.scalar_tensor(2, device=params['device'])

    def forward(self, x, t, f, s, s_trunk, z_trunk):
        #conditioning
        s, z_trunk = self.diffusion_conditioning(t, f, s, s_trunk, z_trunk)
                    

        #scale positions to dimensionbless vectors with approximatley unit variance
        r = x / torch.sqrt(torch.square(t) + torch.square(self.sigma_data))


        # Sequence-local atom attention adn aggregation to coarse grain tokens
        a, q, c, p = self.atom_attention_encoder(f, s_trunk, z_trunk, r) # needs inprovement


        a += self.s_inp_proj(self.s_inp_norm(s))
        a = self.diffusion_transformer(a, s, z_trunk, beta=0) #Runs beautifully!!!!


        #Full attention on token level
        a = self.diff_norm(a)
        r = self.atom_attention_decoder(a, q, c, p, f['reference_tokens'])
        #Broadcast token activation to atoms and run sequence local atom attention

        x = torch.square(self.sigma_data) / (torch.square(self.sigma_data) + torch.square(t))*x + self.sigma_data*t / torch.sqrt(torch.square(self.sigma_data) + torch.square(t)) *  r  #could make this a kernal



        #rescale updates to positions and combine with input positions

        return x  

# @torch.jit.script #creates a kernal to make this a single opperation rather than 2 or 3 idk if 1+ counts
# def get_noise(noise_scaler_lambda, t, noise_coef, batch_size, natoms, device):
#     return noise_scaler_lambda * torch.sqrt(torch.square(t) + torch.square(noise_coef)) * torch.rand((batch_size, natoms), device=device)

class Diffuser(nn.Module): # NOTE: Algo 18 : Sample Diffusion
    def __init__(self, params):
        super(Diffuser, self).__init__()

        self.device = params['device']

        self.catoms = params['catoms']
        self.strans = torch.scalar_tensor(params['strans'], device=self.device)
        self.gamma_0 = torch.scalar_tensor(params['gamma_0'], device=self.device)
        self.gamma_min = torch.scalar_tensor(params['gamma_min'], device=self.device)
        self.noise_scaler_lambda = torch.scalar_tensor(params['noise_scale_lambda'], device=self.device)
        self.step_scale_eta = torch.scalar_tensor(params['step_scale_eta'], device=self.device)
        self.noise_scheular = NoiseSchedular(params['timesteps'], self.device)
        self.diff_model = DiffusionModule(params)
        self.central_rand_aug = CentreRandomAugmentation(self.strans, self.device)

        self.zero = torch.scalar_tensor(0, device=self.device)
        self.one = torch.scalar_tensor(1, device=self.device)

    def forward(self, f, s_inp, s_trunk, z_trunk):
        
        # N = torch.distributions.multivariate_normal.MultivariateNormal(torch.zeros(3, device=self.device), torch.eye(3, device=self.device))
        # x = N.sample((s_inp.shape[0], self.Natoms))
        x = torch.randn((s_inp.shape[0], f['ref_pos'].shape[1], 3), device=self.device)
        for i, ct in enumerate(self.noise_scheular.posterior_varience[1:]):
            x = self.central_rand_aug(x)
            gamma = self.gamma_0 if ct > self.gamma_min else self.zero
            t = self.noise_scheular.posterior_varience[i+1]*(gamma + self.one) # i is +1 to the real idx
            
            ei = self.noise_scaler_lambda * torch.sqrt(torch.square(t) + torch.square(self.noise_scheular.posterior_varience[i])) * torch.rand((s_inp.shape[0], f['ref_pos'].shape[1], 3), device=self.device)
            x_noisy = x + ei
            x_denoised = self.diff_model(x_noisy, t, f, s_inp, s_trunk, z_trunk) #main bottle neck
            delta = (x - x_denoised) / t

            dt = ct - t
            x = x_noisy + self.step_scale_eta * dt * delta

            # ei = get_noise(self.noise_scaler_lambda, t, self.noise_scheular.posterior_varience[i], s_inp.shape[0], self.Natoms, self.device
        return x
    
class InputFeatureEmbedder(nn.Module):
    def __init__(self, params): 
        super(InputFeatureEmbedder, self).__init__()
        #catoms=128, catompairs=16, ctoken=384, nblocks=3, nheads=4, nqueries=32, nkeys=128, transition_n=2
        self.AAE = AtomAttentionEncoder(params, ctokens=params['AAE_ctoken']) 


    def forward(self, f):
        a, _, _, _ = self.AAE(f, None, None, r=None)
        s = torch.concat(
            [a, f['restype']], dim=-1)
        return s

class ConfidenceHead(nn.Module):
    def __init__(self, params):
        super(ConfidenceHead, self).__init__()
        
        cs = params['cs']
        cz = params['cz']
        pae_c = params['pae_c']
        pde_c = params['pde_c']
        pres_c = params['pres_c']
        plddt_c = params['plddt_c']

        self.pairformer = Pairformer(params)

        self.z_proj1 = nn.Linear(cs, cz, bias=False)
        self.z_proj2 = nn.Linear(cs, cz, bias=False)
        
        self.d_proj = nn.Linear(1, cz, bias=False)

        self.PAE_proj = nn.Linear(cz, pae_c)
        self.PDE_proj = nn.Linear(cz, pde_c)
        self.PLDDT_proj = nn.Linear(cs, plddt_c)
        self.PRES_proj = nn.Linear(cs, pres_c)


    def forward(self, s_inp, s, z, x, representative_atoms, reference_tokens=None):
        
        """
        Inputs : 
            s_inp   : b x i x cs 
            s       : b x i x 
            z       : b x i x i x cz
            x       : b x l x 3
            representative_atoms : b x i x l
        
        """

        z += self.z_proj1(s_inp).unsqueeze(1) + self.z_proj2(s_inp).unsqueeze(2)
        x = extract_representative_atoms(x, representative_atoms)
        d = torch.abs((x.unsqueeze(1) - x.unsqueeze(2)).sum(-1, keepdim=True))
        z += self.d_proj(d)

        s_add, z_add = self.pairformer(s, z)
        s += s_add
        z += z_add
        
        p_pae = F.softmax(
            self.PAE_proj(z), dim=-1
        )
        p_pde = F.softmax(
            self.PDE_proj(z+z.permute((0,2,1,3))), dim=-1
        )
        p_plddt = F.softmax(
            self.PLDDT_proj(s)
            , dim=-1 )
        p_res = F.softmax(
            self.PRES_proj(s)
        , dim=-1 )
        return p_pae, p_pde, p_plddt, p_res


class AlphaFold3_TEMPLE(nn.Module):
    def __init__(self, params):
        super(AlphaFold3_TEMPLE, self).__init__()

        self.Ncycles = params['N_cycles']
        s_dim = params['s_dim']
        cs = params['cs']
        cz = params['cz']

        self.input_features_embed = InputFeatureEmbedder(params)
        self.pairformer = Pairformer(params) 
        self.rope = RelativePositionEncoding(params)
        self.diffusion = Diffuser(params)
        self.confidence_head = ConfidenceHead(params)


        self.s_init_proj = nn.Linear(s_dim, cs, bias=False)
        self.z_init_proj = nn.Linear(cs, cz, bias=False)

        self.token_bonds_proj = nn.Linear(1,cz, bias=False)

        self.z_proj = nn.Linear(cz, cz, bias=False)
        self.z_norm = nn.LayerNorm(cz)

        self.s_proj = nn.Linear(cs, cs, bias=False)
        self.s_norm = nn.LayerNorm(cs)




    def forward(self, f):
        s_init = self.input_features_embed(f)

        s_init = self.s_init_proj(s_init)
        
        z_init = self.z_init_proj(s_init) # b,n,c
        z_init = z_init.unsqueeze(1) + z_init.unsqueeze(2) #b,1,n,c + b,n,1,c -> b, n, n, c

        z_init += self.rope(f)
        z_init += self.token_bonds_proj(f['token_bonds'].unsqueeze(3))

        z = torch.zeros_like(z_init)
        s = torch.zeros_like(s_init)

        for i in range(self.Ncycles):
            z = z_init + self.z_proj(self.z_norm(z))
            s = s_init + self.s_proj(self.s_norm(s))
            s, z = self.pairformer(s, z)

        x = self.diffusion(f, s_init, s, z)

        p_pae, p_pde, p_plddt, p_res = self.confidence_head(s_init, s, z, x, f['represenative_atoms'])
        return x, p_pae, p_pde, p_plddt, p_res

def test_preformer_layer():
    if torch.cuda.is_available():
        print("Found Cuda")
        device = torch.device("cuda")
    else:
        print("No GPU found")
        device = torch.device("cpu")
    print(f"Pytorch using device : {device}")
    
    batch_size = 3
    sequence_size = 10
    channel_size = 32

    m = PairformerLayer(channel_size).to(device)
    
    a = torch.randn(batch_size, sequence_size, channel_size, device=device)
    b = torch.randn(batch_size, sequence_size, sequence_size, channel_size, device=device)
    z, s = m(a,b)
    print(f"z shape : {z.shape}, s : {s.shape}")

def test_AF3_MSALess():
    if torch.cuda.is_available():
        print("Found Cuda")
        device = torch.device("cuda")
    else:
        print("No GPU found")
        device = torch.device("cpu")
    print(f"Pytorch using device : {device}")
    
    batch_size = 2
    nresidues = 5
    ntokens = 5
    natoms = 10
    nbonds = 5

    catom = 128
    catompair = 16
    ctokens = 384

    s_dim = ctokens + 32
    a_dim = ctokens
    f_sum_sum_AAE = None

    res_one_hot =  torch.zeros((batch_size, ntokens, 32))
    res_one_hot[:, :, 0] = 1
    
    atom_one_hot =  torch.zeros((batch_size, natoms, 128))
    atom_one_hot[:, :, 0] = 1

    atom_char_one_hot = torch.zeros((batch_size, natoms, 4, 64))
    atom_char_one_hot[:, :, :, 0] = 1

    f = {
        "residue_index" : torch.zeros((batch_size, ntokens)), # Residues number in the token's original input chain
        "token_index" : torch.zeros((batch_size, ntokens)), # Token number. increases monotonically does no restart at 1 for new chains
        "asym_id" : torch.zeros((batch_size, ntokens)), # Unique integer for each distinct chain
        "entity_id" : torch.zeros((batch_size, ntokens)), # Unique integer for each distinct sequence
        "sym_id" : torch.zeros((batch_size, ntokens)), # Unique integer within chains of this sequence. E.g. if chains A, B and C share a sequence but D does not, their sym_ids would be [0, 1, 2, 0].
        "restype" : res_one_hot, #One-hot encoding of the sequence. 32 possible values: 20 amino acids + unknown, 4 RNA nucleotides + unknown, 4 DNA nucleotides + unknown, and gap. Ligands represented as “unknown amino acid”.
        "is_protein" :torch.zeros((batch_size, ntokens)), # Mask for a molecule type
        "is_rna" : torch.zeros((batch_size, ntokens)),# Mask for a molecule type
        "is_dna" : torch.zeros((batch_size, ntokens)),# Mask for a molecule type
        "is_ligand" : torch.zeros((batch_size, ntokens)),# Mask for a molecule type
        "ref_pos" : torch.randn((batch_size, natoms, 3)), #Atom positions in the reference conformer, with a random rotation and translation applied. Atom positions are given in Å.
        "ref_mask" : torch.zeros((batch_size, natoms)), # Mask indicating which atom slots are used in the reference conformer.
        "ref_charge" : torch.zeros((batch_size, natoms)), #Charge for each atom in the reference conformer.
        "ref_element" : atom_one_hot, #One-hot encoding of the element atomic number for each atom in the reference conformer, up to atomic number 128.
        "ref_atom_name_chars" : atom_char_one_hot, # One-hot encoding of the unique atom names in the reference conformer. Each character is encoded as ord(c) − 32, and names are padded to length 4
        "ref_space_uid" :  torch.zeros((batch_size, natoms)), # Numerical encoding of the chain id and residue index associated with this reference conformer. Each (chain id, residue index) tuple is as- signed an integer on first appearance.
        "token_bonds" : torch.zeros(batch_size, ntokens, ntokens)

    }

    Ntokens = f['residue_index'].shape[1]
    Natoms = f['ref_pos'].shape[1]

    params ={
        "s_dim" : ctokens + 32,
        "Ntokens" : Ntokens,
        "Natoms" : Natoms,
        "rmax" : 32,
        "smax" : 2,
        "AAE_f_sum_dim" : 389, 
        "AAE_nblocks" : 1, # was 3
        "AAE_nheads" : 4,
        "AAE_nqueries" : 32,
        "AAE_mkeys" : 128,
        "AAE_transition_n" : 2,
        "AAE_ctoken" : 384,
        "AAD_nheads" : 4,
        "AAD_nblocks" : 1, #3
        "IFE_s_size" : 384 + 32,
        "AT_nblock" : 1, # was 3,
        "AT_nheads" : 4,
        "AT_nqueries" : 32,
        "AT_nkeys" : 128,
        "S_subset_centres" : [15.5, 47.5, 79.5],
        "cz" : 128,
        "cs" : 384,
        "ca" : 384,
        "DM_ctokens" : 768,
        "DM_DT_nblocks" : 1, #24,
        "DM_DT_nheads" : 16,
        "DT_transition_scale" : 2,
        "catom_pairs" : 16,
        "catoms" : 128,
        "signma_data" : 16,
        "gamma_0" : 0.8,
        "gamma_min" : 1.0,
        "noise_scale_lambda" : 1.003,
        "step_scale_eta" : 1.5,
        "strans" : 1,
        "timesteps" : 20,
        "Pairformer_nblocks" : 1, # was 48,
        "Pairformer_pair_nheads" : 16,
        "Pairformer_drop_p" : 0.25,
        "Pairformer_transition_scale" : 4,
        "Triangle_attention_channel" : 32,
        "Triangle_attention_nheads" : 4,
        "Triangle_multi_c" : 128,
        "pae_c" : 64,
        "pde_c" : 64,
        "plddt_c" : 50,
        "pres_c" : 2,
        "Conf_Head_nblocks" : 4,
        "Fourier_n" : 256,
        "N_cycles" : 1,
        "device" : device
    }
    AF3 = AlphaFold3_TEMPLE(params)
    with torch.no_grad():
        AF3(f)

if __name__ == "__main__":
    # test_preformer_layer()
    test_AF3_MSALess()
