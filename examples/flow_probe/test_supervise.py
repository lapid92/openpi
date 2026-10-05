"""Safety tests for publication sharding and immutable smoke gate."""
import hashlib,json,tempfile,unittest
from pathlib import Path
from examples.flow_probe.supervise import shard_records,validate_smoke_gate

class SupervisorTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix="test-supervise-",dir="/volt/artifacts/flow-probe/test-writer")
        self.root=Path(self.temp.name)
    def tearDown(self):self.temp.cleanup()
    def test_lossless_boundaries(self):
        p=self.root/"raw.jsonl";raw=b'{"a":1}\n{"b":2}\n{"c":3}';p.write_bytes(raw)
        dst=self.root/"shards";r=shard_records(p,dst,limit=16)
        self.assertEqual(b"".join((dst/x["path"]).read_bytes() for x in r["shards"]),raw)
        self.assertEqual(r["source_sha256"],hashlib.sha256(raw).hexdigest())
        self.assertEqual(r["lines"],3)
        self.assertTrue(all(x["bytes"]<=16 for x in r["shards"]))
    def test_existing_shards_refused(self):
        p=self.root/"raw.jsonl";p.write_text('{"a":1}\n');dst=self.root/"shards"
        shard_records(p,dst)
        with self.assertRaises(FileExistsError):shard_records(p,dst)
    def test_oversized_row_refused(self):
        p=self.root/"raw.jsonl";p.write_text('{"verylong":123456789}\n')
        with self.assertRaises(RuntimeError):shard_records(p,self.root/"shards",limit=5)
    def fixture(self):
        m={"benchmarks":{"libero":{},"libero_plus":{}}};mp=self.root/"manifest.json";mp.write_text(json.dumps(m))
        h=hashlib.sha256(mp.read_bytes()).hexdigest();gate={"status":"passed","manifest_sha256":h}
        for field,names in {
          "record_files":[b+"-smoke.jsonl" for b in m["benchmarks"]],
          "metrics_files":[b+"-smoke-metrics.json" for b in m["benchmarks"]],
          "preparation_files":[b+"-smoke-prepare-worker-"+str(i)+".jsonl" for b in m["benchmarks"] for i in range(4)]}.items():
            gate[field]=[]
            for name in names:
                p=self.root/name;p.write_text("{}\n")
                gate[field].append({"path":str(p),"sha256":hashlib.sha256(p.read_bytes()).hexdigest()})
        (self.root/"smoke-status.json").write_text(json.dumps({"status":"published","manifest_sha256":h}))
        (self.root/"smoke-independent-audit.json").write_text(json.dumps(gate))
        return m,mp,gate
    def test_gate_exact_evidence_and_tamper(self):
        m,mp,g=self.fixture();validate_smoke_gate(m,mp,self.root)
        Path(g["record_files"][0]["path"]).write_text('{"changed":true}\n')
        with self.assertRaises(RuntimeError):validate_smoke_gate(m,mp,self.root)
    def test_gate_missing_preparation_refused(self):
        m,mp,g=self.fixture();g["preparation_files"].pop()
        (self.root/"smoke-independent-audit.json").write_text(json.dumps(g))
        with self.assertRaises(RuntimeError):validate_smoke_gate(m,mp,self.root)
    def test_gate_unpublished_refused(self):
        m,mp,g=self.fixture()
        (self.root/"smoke-status.json").write_text(json.dumps({"status":"completed","manifest_sha256":g["manifest_sha256"]}))
        with self.assertRaises(RuntimeError):validate_smoke_gate(m,mp,self.root)

if __name__=="__main__":unittest.main()

