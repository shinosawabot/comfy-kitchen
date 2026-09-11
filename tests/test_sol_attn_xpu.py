"""Public XPU dispatch and compile contracts; all numerical oracles run on XPU."""
import pytest
import torch
import comfy_kitchen as ck

pytestmark=pytest.mark.skipif(not ck.sol_attn_is_available(torch.device("xpu")),
                            reason="complete native XPU Sol sidecar unavailable")


@pytest.mark.parametrize("dtype",[torch.bfloat16,torch.float16])
@pytest.mark.parametrize("budget",[0,64,256])
@pytest.mark.parametrize("bias_kind",["none","float","bool"])
def test_public_exact_sink_output_and_dispatch(dtype,budget,bias_kind,monkeypatch):
    gen=torch.Generator(device="xpu").manual_seed(20147)
    q,k,v=torch.randn((2,257,3,3,128),device="xpu",dtype=dtype,generator=gen).unbind(2)
    bias=None
    if bias_kind=="float":bias=torch.randn((2,1,1,257),device="xpu",generator=gen)*0.2
    elif bias_kind=="bool":bias=torch.arange(257,device="xpu")%7!=0
    reference_bias=bias
    if bias_kind=="bool":reference_bias=torch.where(bias,0.0,float("-inf"))
    score=q.permute(0,2,1,3).float() @ k.permute(0,2,3,1).float()*128**-0.5
    if reference_bias is not None:score+=reference_bias
    wanted=(torch.softmax(score,-1) @ v.permute(0,2,1,3).float()).permute(0,2,1,3)
    calls=[]
    native=ck._xpu_backend.sol_attn
    def observe(**kwargs):
        calls.append(kwargs)
        return native(**kwargs)
    monkeypatch.setattr(ck._xpu_backend,"sol_attn",observe)
    out=ck.sol_attn(q,k,v,sink_blocks=[0,99],key_bias=bias,token_aug=budget)
    assert len(calls)==1 and calls[0]["token_aug"]==budget
    assert out.is_contiguous() and out.dtype==dtype and out.shape==q.shape
    torch.testing.assert_close(out.float(),wanted,rtol=0.02,atol=0.02)


@pytest.mark.parametrize("tail",[False,True])
@pytest.mark.parametrize("topk",[0.0,0.3])
@pytest.mark.parametrize("budget",[0,64,256])
def test_public_sparse_controls_reach_native(tail,topk,budget,monkeypatch):
    gen=torch.Generator(device="xpu").manual_seed(20219)
    q,k,v=torch.randn((1,577,3,3,128),device="xpu",dtype=torch.bfloat16,generator=gen).unbind(2)
    lengths=torch.tensor([61,57,52,48,39,63,47,59,60,1],device="xpu",dtype=torch.int32)
    gate=torch.randn(q.shape,device="xpu",generator=gen)*0.1
    native=ck._xpu_backend.sol_attn
    kwargs=dict(tau=1.3,topk_ratio=topk,tail=tail,block_len=lengths,
                coarse_gate=gate,token_aug=budget,sink_blocks=[0,1],sink_q=[0,1])
    expected=native(q,k,v,**kwargs)
    calls=[]
    def observe(**arguments):
        calls.append(arguments)
        return native(**arguments)
    monkeypatch.setattr(ck._xpu_backend,"sol_attn",observe)
    out=ck.sol_attn(q,k,v,**kwargs)
    assert len(calls)==1 and calls[0]["tail"]==tail and calls[0]["topk_ratio"]==topk
    assert calls[0]["coarse_gate"] is gate and calls[0]["block_len"] is lengths
    torch.testing.assert_close(out,expected,rtol=0,atol=0)


def test_public_custom_op_compile_shape_and_dtype():
    q,k,v=torch.randn((1,65,3,3,128),device="xpu",dtype=torch.bfloat16).unbind(2)
    def run(q,k,v):return ck.sol_attn(q,k,v,sink_blocks=[0,2],token_aug=64)
    compiled=torch.compile(run,backend="eager",fullgraph=True)
    output=compiled(q,k,v)
    torch.testing.assert_close(output,run(q,k,v),rtol=0,atol=0)
    assert output.is_contiguous()


@pytest.mark.parametrize("bootstrap",[False,True])
def test_public_chunked_factory_replay_and_input_ownership(bootstrap):
    t,h=257,3;gen=torch.Generator(device="xpu").manual_seed(20333)
    packed=torch.randn((t,3*h*128),device="xpu",dtype=torch.bfloat16,generator=gen)
    before=packed.clone();calls=[]
    def factory():
        calls.append(1)
        return (packed[i:i+128] for i in range(0,t,128))
    freqs=torch.eye(2,device="xpu",dtype=torch.bfloat16).expand(1,t,1,48,2,2).contiguous()
    weights=(torch.ones(128,device="xpu",dtype=torch.bfloat16),)*2
    km=None if bootstrap else torch.zeros((h,128),device="xpu")
    vs=None if bootstrap else torch.full((h,128),0.1,device="xpu")
    out,next_k,next_v=ck.sol_attn_chunked(factory,t,h,freqs,weights,km,vs,
                                       sink_blocks=[0,5],token_aug=256)
    assert len(calls)==(2 if bootstrap else 1)
    torch.testing.assert_close(packed,before,rtol=0,atol=0)
    assert out.shape==(1,t,h,128) and out.dtype==torch.bfloat16 and out.is_contiguous()
    assert next_k.shape==next_v.shape==(h,128)
    assert next_k.dtype==next_v.dtype==torch.float32
    q,k,v=packed.view(1,t,3,h,128).unbind(2)
    q=(q.float()*torch.rsqrt(q.float().square().mean(-1,keepdim=True)+1e-6)).to(torch.bfloat16)
    k=(k.float()*torch.rsqrt(k.float().square().mean(-1,keepdim=True)+1e-6)).to(torch.bfloat16)
    score=q.permute(0,2,1,3).float() @ k.permute(0,2,3,1).float()*128**-0.5
    wanted=(torch.softmax(score,-1) @ v.permute(0,2,1,3).float()).permute(0,2,1,3)
    torch.testing.assert_close(out.float(),wanted,rtol=0.02,atol=0.02)


@pytest.mark.parametrize("device",[None,0,torch.device("xpu"),"xpu:0"])
def test_available_device_forms(device):
    assert ck.sol_attn_is_available(device)
