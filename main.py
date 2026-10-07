"""通用视频运动目标跟踪：单文件课程实验。
python main.py --make-demo
python main.py --source data/demo.avi
python main.py --source 0
"""
import argparse
import csv
from pathlib import Path
from collections import deque
from math import hypot
import cv2
import numpy as np

# ==================== 预处理 ====================
"""预处理：对应实验2亮度加减、实验5滤波与卷积。"""


def preprocess(frame, brightness=20, sharpen=True):
    # 灰度表示简化后续计算；固定亮度偏移避免逐帧均衡化引起背景变化。
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corrected = cv2.add(gray, brightness) if brightness >= 0 else cv2.subtract(gray, -brightness)
    blurred = cv2.GaussianBlur(corrected, (5, 5), 0)
    # 温和的拉普拉斯锐化卷积：中心增强、相邻像素相减。
    kernel = np.array([[0, -.2, 0], [-.2, 1.8, -.2], [0, -.2, 0]], dtype=np.float32)
    enhanced = cv2.filter2D(blurred, -1, kernel) if sharpen else blurred.copy()
    return corrected, blurred, enhanced

# ==================== 分割与目标表示 ====================
"""背景差分与形态学：对应实验2差运算、实验6阈值与开闭运算。"""


class Segmenter:
    def __init__(self, threshold=25):
        self.background = None
        self.threshold = threshold

    def reset(self, image):
        # 参考背景固定不更新；采集时画面应没有待跟踪目标。
        self.background = image.copy()

    def process(self, image):
        if self.background is None:
            self.reset(image)
        difference = cv2.absdiff(image, self.background)
        _, binary = cv2.threshold(difference, self.threshold, 255, cv2.THRESH_BINARY)
        opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
        return difference, binary, opened, closed


def detect(mask, min_area=400):
    # 将分割区域表示为轮廓、外接矩形、质心和面积。
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detections = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue
        moments = cv2.moments(contour)
        if moments['m00'] == 0:
            continue
        center = (int(moments['m10']/moments['m00']), int(moments['m01']/moments['m00']))
        detections.append({'center': center, 'box': cv2.boundingRect(contour), 'area': area})
    return detections

# ==================== 质心跟踪 ====================
"""基础质心跟踪：距离计算与一对一匹配，不使用学习模型。"""


class CentroidTracker:
    def __init__(self, max_distance=80, max_missing=10):
        self.max_distance = max_distance
        self.max_missing = max_missing
        self.tracks = {}
        self.next_id = 1

    def update(self, detections):
        # 全部候选按距离排序，贪心匹配；每个旧目标和检测仅使用一次。
        candidates = []
        for identity, track in self.tracks.items():
            for index, detection in enumerate(detections):
                a, b = track['center'], detection['center']
                distance = hypot(a[0]-b[0], a[1]-b[1])
                if distance <= self.max_distance:
                    candidates.append((distance, identity, index))
        used_tracks, used_detections = set(), set()
        for _, identity, index in sorted(candidates):
            if identity in used_tracks or index in used_detections:
                continue
            track = self.tracks[identity]
            track.update(detections[index])
            track['missing'] = 0
            track['trail'].append(track['center'])
            used_tracks.add(identity)
            used_detections.add(index)
        for identity in list(self.tracks):
            if identity not in used_tracks:
                self.tracks[identity]['missing'] += 1
                if self.tracks[identity]['missing'] > self.max_missing:
                    del self.tracks[identity]
        for index, detection in enumerate(detections):
            if index not in used_detections:
                self.tracks[self.next_id] = dict(detection, missing=0, trail=deque([detection['center']], maxlen=40))
                self.next_id += 1
        # 漏检目标暂时保留用于重关联，但不在当前画面绘制旧框。
        return {identity: track for identity, track in self.tracks.items() if track['missing'] == 0}

# ==================== 多窗口与绘图 ====================
"""九个独立窗口展示流程；窗口标题用英文避免OpenCV中文乱码。"""

TITLES = ['01 Original', '02 Brightness', '03 Gaussian', '04 Sharpen',
          '05 Difference', '06 Threshold', '07 Opening', '08 Closing', '09 Tracking']


def draw_tracks(frame, tracks):
    result = frame.copy()
    for identity, track in tracks.items():
        color = (80 + identity*53 % 176, 80 + identity*97 % 176, 80 + identity*31 % 176)
        x, y, w, h = track['box']
        cv2.rectangle(result, (x, y), (x+w, y+h), color, 2)
        cv2.circle(result, track['center'], 4, color, -1)
        cv2.putText(result, f'ID {identity}', (x, max(y-6, 18)), cv2.FONT_HERSHEY_SIMPLEX, .6, color, 2)
        if len(track['trail']) > 1:
            cv2.polylines(result, [np.array(track['trail'], np.int32)], False, color, 2)
    cv2.putText(result, f'Targets: {len(tracks)} | B: background SPACE: pause Q: quit',
                (10, 24), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 255, 0), 1)
    return result


class Display:
    def __init__(self):
        for i, title in enumerate(TITLES):
            cv2.namedWindow(title, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(title, 400, 225)
            cv2.moveWindow(title, (i % 3)*410, (i // 3)*260)

    def show(self, images):
        for title, image in zip(TITLES, images):
            cv2.imshow(title, image)

    def closed(self):
        return any(cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) < 1 for title in TITLES)

# ==================== 演示视频生成 ====================
def make_demo():
    folder = Path(__file__).parent / 'data'
    folder.mkdir(exist_ok=True)
    writer = cv2.VideoWriter(str(folder/'demo.avi'), cv2.VideoWriter_fourcc(*'MJPG'), 25, (800, 450))
    if not writer.isOpened():
        raise RuntimeError('Cannot create demo video')
    try:
        for i in range(200):
            frame = np.full((450, 800, 3), 45, np.uint8)
            if 25 <= i < 175:
                x = 30 + (i-25)*4
                cv2.rectangle(frame, (x, 100), (x+65, 160), (180, 200, 230), -1)
                cv2.circle(frame, (730-(i-25)*4, 310), 30, (200, 180, 160), -1)
            writer.write(frame)
    finally:
        writer.release()
    print(folder/'demo.avi')



# ==================== 输入与主流程 ====================
def main():
    parser = argparse.ArgumentParser(description='Basic motion target tracking')
    parser.add_argument('--make-demo', action='store_true', help='Generate demo and exit')
    parser.add_argument('--source', default='0', help='Video path or camera index')
    parser.add_argument('--brightness', type=int, default=20)
    parser.add_argument('--threshold', type=int, default=25)
    parser.add_argument('--min-area', type=float, default=400)
    parser.add_argument('--max-distance', type=float, default=80)
    parser.add_argument('--width', type=int, default=800)
    parser.add_argument('--no-sharpen', action='store_true')
    parser.add_argument('--save', action='store_true', help='Save annotated video and CSV')
    parser.add_argument('--headless', action='store_true', help='Run without display for validation')
    args = parser.parse_args()
    if args.make_demo:
        make_demo()
        return
    if not 0 <= args.threshold <= 255 or not -255 <= args.brightness <= 255:
        parser.error('Threshold must be 0..255; brightness must be -255..255')
    if args.width < 1 or args.min_area <= 0 or args.max_distance <= 0:
        parser.error('Width, min-area and max-distance must be positive')
    source = int(args.source) if args.source.isdecimal() else args.source
    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        capture.release()
        raise SystemExit(f'Cannot open source: {args.source}')
    segmenter = Segmenter(args.threshold)
    tracker = CentroidTracker(args.max_distance)
    writer = csv_file = None
    display = None
    frame_index = 0
    fps = capture.get(cv2.CAP_PROP_FPS)
    if not 1 <= fps <= 240:
        fps = 25
    try:
        display = None if args.headless else Display()
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            # 统一宽度：面积阈值和匹配距离均以处理后的像素为单位。
            height = max(1, round(frame.shape[0]*args.width/frame.shape[1]))
            frame = cv2.resize(frame, (args.width, height))
            corrected, blurred, enhanced = preprocess(frame, args.brightness, not args.no_sharpen)
            diff, binary, opened, closed = segmenter.process(enhanced)
            tracks = tracker.update(detect(closed, args.min_area))
            result = draw_tracks(frame, tracks)
            frame_index += 1
            if args.save:
                if writer is None:
                    output = Path(__file__).parent / 'output'
                    output.mkdir(exist_ok=True)
                    writer = cv2.VideoWriter(str(output/'tracking.avi'), cv2.VideoWriter_fourcc(*'MJPG'), fps, (args.width, height))
                    if not writer.isOpened():
                        raise RuntimeError('Video encoder could not open')
                    csv_file = (output/'tracks.csv').open('w', newline='', encoding='utf-8-sig')
                    rows = csv.writer(csv_file)
                    rows.writerow(['frame', 'id', 'cx', 'cy', 'x', 'y', 'w', 'h', 'area'])
                writer.write(result)
                for identity, track in tracks.items():
                    rows.writerow([frame_index, identity, *track['center'], *track['box'], track['area']])
            if display:
                display.show([frame, corrected, blurred, enhanced, diff, binary, opened, closed, result])
                key = cv2.waitKey(max(1, round(1000/fps))) & 255
                if key == 32:  # 暂停后按任意键继续，也接受退出/背景重设键。
                    key = cv2.waitKey(0) & 255
                if key in (27, ord('q')) or display.closed():
                    break
                if key == ord('b'):
                    segmenter.reset(enhanced)
                    tracker = CentroidTracker(args.max_distance)
        print(f'Processed {frame_index} frames')
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        if csv_file is not None:
            csv_file.close()
        if display is not None:
            cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
