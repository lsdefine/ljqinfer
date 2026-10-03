"""Lazy random values in canonical EP8/TP8 layout; no computation mocking."""
import torch
from released_random import MixedWeights
from model.weights import shard


class BankView:
    def __init__(self, owner, name):
        self.owner, self.name = owner, name

    def __getitem__(self, local):
        owner = self.owner
        if not 0 <= local < 48:
            raise ValueError('invalid local expert')
        base, suffix = self.name.split('.local_experts.')
        which, kind = suffix.split('.')
        key = (base,local,which)
        if kind == 'scale':
            if owner.bank_scale is None or owner.bank_scale[0] != key:
                raise RuntimeError('bank scale must follow weight')
            scale = owner.bank_scale[1]
            owner.bank_scale = None
            return scale
        expert = owner.rank*48+local
        values, scales = [], []
        for projection in (('w1','w3') if which == 'w13' else ('w2',)):
            p = f'{base}.experts.{expert}.{projection}'
            values.append(owner.full[p+'.weight'])
            scales.append(owner.full[p+'.scale'])
        owner.experts.add(expert)
        value = torch.stack(values) if which == 'w13' else values[0]
        scale = torch.stack(scales) if which == 'w13' else scales[0]
        owner.bank_scale = (key,scale)
        return value


class RankRandom:
    def __init__(self, device, rank):
        self.full, self.rank = MixedWeights(device), rank
        self.bank_scale = None
        self.experts = set()

    def __getitem__(self, name):
        if '.local_experts.' in name:
            return BankView(self,name)
        return shard(name,self.full[name],self.rank)
