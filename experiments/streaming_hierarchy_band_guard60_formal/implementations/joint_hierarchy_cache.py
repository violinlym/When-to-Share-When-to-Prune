"""Experiment7: strict-budget, causal, irreversible native joint hierarchy."""
import time
import numpy as np
import torch
from transformers.cache_utils import DynamicCache,DynamicLayer
from joint_cache import JointCache,JointLayer
from hierarchy import build_hierarchy
from heap_allocator import allocate,masks


class JointHierarchyLayer(JointLayer):
    def tree(self,kv):
        if kv.device.type=='cuda':torch.cuda.synchronize(kv.device)
        start=time.perf_counter()
        descending,order=build_hierarchy(kv)
        self.pending_cost=descending.flip(-1)
        self.pending_order=order.flip(-1)
        if kv.device.type=='cuda':torch.cuda.synchronize(kv.device)
        self.tree_seconds+=time.perf_counter()-start

    def initialize_statistics(self,kv,scores):
        self.tree_seconds=0.;self.tree(kv)
        self.mass=scores.float().mean(dim=0)

    def update_statistics(self):
        self.tree(self.current)
        assert len(self.attention_weights)==4
        delta=torch.stack(self.attention_weights).float().mean(dim=0)
        self.mass=torch.cat((self.mass,torch.zeros_like(self.mass[...,:1])),dim=-1)+delta
        self.attention_weights=[]


class JointHierarchyCache(JointCache):
    def __init__(self,source,mc,ratio,retention=.5):
        DynamicCache.__init__(self)
        assert ratio==.25 and mc['loops']==4 and mc['kv_heads']==mc['query_heads']
        assert retention in (.5,1.)
        self.retention=retention
        self.L,self.R,self.H=mc['layers'],4,mc['kv_heads'];self.groups=1
        self.layers=[DynamicLayer() for _ in range(self.L*self.R)]
        self.method,self.ratio='joint_hierarchy',ratio
        self.n=self.prompt_tokens=source['prompt_tokens'];self.device=source['kv'][0].device
        self.calls=self.attention_calls=self.steps=0
        self.decision_seconds=self.statistics_seconds=0.
        self.heap_seconds=self.scoring_seconds=self.transfer_mask_seconds=self.compaction_seconds=0.
        self.costs=np.empty((0,2,3),np.float32);self.order=np.empty((0,2,3),np.uint8)
        self.stage=np.empty((0,2),np.uint8);self.alive=np.empty(0,np.uint8)
        self.action_counters=np.zeros(16,np.uint64);self.actions_by_head=np.zeros((self.L*self.H,3),np.uint64)
        self.heap_peak_bytes=0;self.cpu_work_peak_bytes=0;self.initial_examples=[]
        self.synchronize();start=time.perf_counter()
        self.states=[JointHierarchyLayer(kv,score,'combined') for kv,score in zip(source['kv'],source['scores'])]
        stat=time.perf_counter()
        for state,kv,score in zip(self.states,source['kv'],source['scores']):state.initialize_statistics(kv,score)
        self.synchronize();self.statistics_seconds+=time.perf_counter()-stat
        self.allocate();self.synchronize();self.allocation_seconds=time.perf_counter()-start

    def allocate(self):
        self.synchronize();start=time.perf_counter()
        # Only newborn native vectors are used for tree construction. Transfer
        # their six scalar merge costs/IDs once; old native versions stay absent.
        cost=torch.stack([s.pending_cost for s in self.states]).permute(3,0,2,1,4).contiguous().cpu().numpy()
        order=torch.stack([s.pending_order for s in self.states]).permute(3,0,2,1,4).contiguous().cpu().numpy()
        new=cost.shape[0]*self.L*self.H
        self.costs=np.concatenate((self.costs,cost.reshape(-1,2,3)))
        self.order=np.concatenate((self.order,order.reshape(-1,2,3)))
        self.stage=np.concatenate((self.stage,np.zeros((new,2),np.uint8)))
        self.alive=np.concatenate((self.alive,np.ones(new,np.uint8)))
        for state in self.states:state.pending_cost=state.pending_order=None
        importance=torch.stack([s.mass/s.n for s in self.states]).permute(2,0,1).contiguous().cpu().numpy().reshape(-1)
        m=self.n*self.L*self.H;assert self.alive.size==m
        target=int(8*m*self.retention)
        prep=time.perf_counter()-start
        result=allocate(importance,self.costs,self.order,self.stage,self.alive,self.n,self.L*self.H,target,
                        allow_prune=self.retention<1,examples=16 if self.steps==0 else 0)
        self.scoring_seconds+=result['scoring_seconds'];self.heap_seconds+=result['heap_seconds']
        self.action_counters+=result['counters'];self.actions_by_head+=result['by_head']
        self.heap_peak_bytes=max(self.heap_peak_bytes,int(result['counters'][7]))
        if self.steps==0:self.initial_examples=result['examples']
        copy_start=time.perf_counter()
        keep_cpu=masks(self.order,self.stage,self.alive,self.n,self.L,self.H)
        keep=torch.from_numpy(keep_cpu).to(self.device)
        self.synchronize();self.transfer_mask_seconds+=prep+time.perf_counter()-copy_start
        self.cpu_work_peak_bytes=max(self.cpu_work_peak_bytes,importance.nbytes+keep_cpu.nbytes+self.heap_peak_bytes)
        self.decision_seconds+=time.perf_counter()-start
        compact=time.perf_counter()
        for state,row in zip(self.states,keep):state.compact(row)
        self.synchronize();self.compaction_seconds+=time.perf_counter()-compact
        assert sum(s.values.shape[0] for s in self.states)==result['remaining']<=target
        protected=(self.n+4)//5
        assert self.alive.reshape(self.n,self.L,self.H)[-protected:].all()

    def memory(self):
        payload=sum(s.values.numel()*s.values.element_size() for s in self.states)
        full=self.L*8*self.H*self.n*self.states[0].D*self.states[0].values.element_size()
        assert payload<=int(full*self.retention)
        assert all(s.current is None and s.n==self.n for s in self.states)
        gpu={name:sum(s.auxiliary()[name] for s in self.states) for name in self.states[0].auxiliary()}
        host={name:getattr(self,name).nbytes for name in ['costs','order','stage','alive','action_counters','actions_by_head']}
        return dict(logical_tokens=self.n,prompt_tokens=self.prompt_tokens,payload_bytes=payload,full_payload_bytes=full,
            payload_ratio=payload/full,auxiliary_bytes=sum(gpu.values())+sum(host.values()),auxiliary_gpu_bytes=sum(gpu.values()),
            auxiliary_host_bytes=sum(host.values()),auxiliary_breakdown_bytes=dict(gpu=gpu,host=host),
            heap_transient_peak_bytes=self.heap_peak_bytes,cpu_decision_transient_peak_bytes=self.cpu_work_peak_bytes,
            numerical_data_dtype='native_bfloat16',target_retention=self.retention,
            counterfactual_attention_forwards=0,sharing_configurations_enumerated=0,
            recent_context_protection=dict(count=(self.n+4)//5,scope='local base roots only'))

    def audit_tree(self):
        native=masks(self.order,self.stage,self.alive,self.n,self.L,self.H)
        assert np.array_equal(native,torch.stack([s.ptr>=0 for s in self.states]).cpu().numpy())
        assert (np.diff(self.costs,axis=-1)>=0).all()
        assert (np.sort(self.order,axis=-1)==np.arange(3)).all()
        return dict(status='PASS',born_roots=int(self.alive.size),live_roots=int(self.alive.sum()),
                    root_deletions=int(self.action_counters[2]),pruning_before_terminal=int(self.action_counters[8]))

    def mechanism(self):
        return dict(tree_seconds=sum(s.tree_seconds for s in self.states),scoring_seconds=self.scoring_seconds,
            heap_seconds=self.heap_seconds,transfer_mask_seconds=self.transfer_mask_seconds,
            compaction_seconds=self.compaction_seconds,importance_seconds=self.statistics_seconds-sum(s.tree_seconds for s in self.states),
            actions_K_sharing=int(self.action_counters[0]),actions_V_sharing=int(self.action_counters[1]),
            actions_pruning=int(self.action_counters[2]),pruning_before_terminal=int(self.action_counters[8]),
            removed_boundaries=self.action_counters[9:15].reshape(2,3).tolist(),
            actions_by_layer_head=self.actions_by_head.reshape(self.L,self.H,3).tolist(),initial_action_examples=self.initial_examples)

    def audit_arrays(self):
        return dict(tree_order=self.order.reshape(self.n,self.L,self.H,2,3).transpose(1,3,2,0,4).copy(),
                    tree_stage=self.stage.reshape(self.n,self.L,self.H,2).transpose(1,3,2,0).copy())
