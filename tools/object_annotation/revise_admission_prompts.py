"""Apply recorded visual boundary corrections without touching scored frames."""
import json

from annotate import BASE, write_json


if __name__ == '__main__':
    source = BASE / 'config/admission_assistant_prompts_20260918.json'
    target = BASE / 'config/admission_assistant_prompts_20260918_r2.json'
    assert not target.exists()
    prompts = json.loads(source.read_text())
    for p in prompts:
        key = (p['case_id'], p['frame_idx'], p['object_id'])
        if key == ('score_T24_head', 189, 4):
            p['points'] += [[292,646], [258,612], [216,540], [422,601], [447,568], [424,663]]
            p['point_labels'] += [0] * 6
            p['note'] = 'Exclude visibly attached left gripper; direct negative prompts, no robot-mask subtraction.'
        elif key == ('score_T24_left_wrist', 355, 10):
            p['box'] = [588,154,640,297]
            p['points'] = [[625,206], [625,252], [637,275], [580,252], [603,306]]
            p['point_labels'] = [1,1,1,0,0]
            p['note'] = 'Include pink blurred peach at right image boundary, not only its pale highlight.'
        elif key == ('score_T24_left_wrist', 355, 4):
            p['points'] += [[625,252], [637,275]]
            p['point_labels'] += [0,0]
            p['note'] = 'Peach boundary exclusion by visible RGB points, not by subtracting another mask.'
    write_json(target, prompts)
    print(target)
