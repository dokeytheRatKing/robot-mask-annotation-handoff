import unittest
import numpy as np
from audit_server import validate_annotation
from masks import encode,record
from evaluate_mask_audit import similarity,aggregate,temporal_metrics,match_identities,evaluate


class AuditTests(unittest.TestCase):
    def test_actual_pixel_metrics(self):
        a=np.zeros((4,4),bool);a[:2,:2]=True;b=a.copy();b[1]=False
        iou,dice=similarity(b,a)
        self.assertAlmostEqual(iou,.5);self.assertAlmostEqual(dice,2/3)
        self.assertEqual(similarity(np.zeros_like(a),a),(0.,0.))
        self.assertEqual(similarity(np.zeros_like(a),np.zeros_like(a)),(None,None))

    def test_visibility_abstention_not_true_negative(self):
        s=aggregate([dict(gt_visible=False,gt_visibility='out_of_view',pred_visible=None,pred_area=0,iou=None,dice=None)])
        self.assertEqual(s['visibility_accuracy'],0);self.assertEqual(s['visibility_abstentions'],1)
        self.assertIsNone(s['mask_iou'])

    def test_reentry_censoring_and_persistence(self):
        seq=[dict(frame_idx=i*5,timestamp=i/6,gt_visibility='out_of_view' if i<2 else 'visible',
                  gt_visible=i>=2,pred_visible=True,iou=0. if i<3 else .9) for i in range(5)]
        fp,re=temporal_metrics(seq)
        self.assertEqual(fp[0]['samples'],2);self.assertTrue(re[0]['success'])
        self.assertAlmostEqual(re[0]['latency_seconds'],1/6)
        self.assertFalse(temporal_metrics(seq[:3])[1][0]['eligible'])
        seq[3]['frame_idx']+=2
        self.assertFalse(temporal_metrics(seq)[1][0]['eligible'])

    def test_identity_matching_exposes_wrong_class(self):
        a=np.zeros((10,10),bool);a[1:4,1:4]=True;b=np.zeros_like(a);b[6:9,6:9]=True
        gt=[dict(object_id=i,instance_id=f'obj{i}',visibility='visible',mask=encode(m)) for i,m in [(1,a),(2,b)]]
        pred=[dict(object_id=i,track_id=str(i),**record(m)) for i,m in [(1,b),(2,a)]]
        matches=match_identities(pred,gt)
        self.assertEqual(len(matches),2)
        self.assertTrue(all(r['pred_object_id']!=r['gt_object_id'] for r in matches))

    def test_no_human_no_quality_claim(self):
        m=dict(samples=[],required_conditions=['grasp_contact'])
        result,_=evaluate(m,[],dict(baseline={},recovery={}))
        self.assertEqual(result['status'],'PENDING_HUMAN_GT')
        self.assertIsNone(result['results']['recovery']['per_camera']['head']['id_switch_count'])
        self.assertIsNone(result['results']['recovery']['overall']['mask_iou'])

    def test_id_switch_count_ignores_segment_restarts(self):
        a=np.zeros((10,10),bool);a[1:4,1:4]=True;b=np.zeros_like(a);b[6:9,6:9]=True
        samples=[];labels=[];swapped={};same={}
        for frame in [0,5]:
            sid=f'frame{frame}';samples.append(dict(sample_id=sid,episode_id='e',camera='head',frame_idx=frame,
                timestamp=frame/30,clip_id='clip',objects=[dict(object_id=1),dict(object_id=2)]))
            for oid,mask in [(1,a),(2,b)]:
                labels.append(dict(sample_id=sid,object_id=oid,instance_id=f'instance{oid}',visibility='visible',mask=encode(mask)))
                key=('e','head',frame,oid)
                same[key]=dict(object_id=oid,track_id=str(oid),segment_id=frame+1,**record(mask))
                swapped[key]=dict(object_id=oid,track_id=str(oid),**record(mask if frame==0 else b if oid==1 else a))
        output,_=evaluate(dict(samples=samples,required_conditions=[]),labels,dict(same=same,swapped=swapped))
        self.assertEqual(output['results']['same']['per_camera']['head']['id_switch_count'],0)
        self.assertEqual(output['results']['swapped']['per_camera']['head']['id_switch_count'],2)
        self.assertEqual(output['results']['swapped']['per_camera']['head']['id_switch_rate'],1.)

    def test_annotation_visibility_and_identity_gate(self):
        sample=dict(sample_id='test',episode_id='test',camera='head',frame_idx=0,
                    image_width=8,image_height=8,image_sha256='fixture',objects=[dict(object_id=1)])
        body=dict(human_confirmed=True,annotator='TEST FIXTURE NOT REAL GT',object_id=1,
                  visibility='out_of_view',instance_id='object_1',mask_rle_counts=[64])
        row=validate_annotation(body,sample,[])
        self.assertEqual(row['visibility'],'out_of_view')
        with self.assertRaises(AssertionError):validate_annotation(body|dict(visibility='visible'),sample,[])
        with self.assertRaises(AssertionError):validate_annotation(body|dict(human_confirmed=False),sample,[])


if __name__=='__main__':unittest.main()
