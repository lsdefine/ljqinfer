import json,unittest
from model.config import CONFIG,validate_model_config
from model.runtime import allocate_mock_cache
class ContractTest(unittest.TestCase):
 def test_source_config(self): validate_model_config()
 def test_geometry(self):
  self.assertEqual(CONFIG.full_attention_layers,tuple(range(3,64,4)))
  self.assertEqual(len(CONFIG.linear_attention_layers),48)
  self.assertEqual(CONFIG.local_q_heads,6); self.assertEqual(CONFIG.local_kv_heads,1)
  self.assertEqual(CONFIG.gdn_conv_state_shape,(3,2560))
  self.assertEqual(CONFIG.gdn_recurrent_state_shape,(12,128,128))
  self.assertEqual(CONFIG.gdn_ssm_dtype,"bfloat16")
 def test_pool(self):
  c=allocate_mock_cache(131072,4,2048)
  self.assertEqual(c.k.shape,(16,64,2048,1,256))
  self.assertEqual(c.v.shape,c.k.shape)
  self.assertEqual(c.gdn_conv.shape,(48,4,3,2560))
  self.assertEqual(c.gdn_recurrent.shape,(48,4,12,128,128))
  self.assertEqual(c.gdn_recurrent.dtype,"bfloat16")
  self.assertEqual(c.k.nbytes+c.v.nbytes,2*1024*1024*1024)
  self.assertEqual(c.page_table.dtype,"int64")
if __name__=='__main__': unittest.main()
