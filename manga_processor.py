#!/usr/bin/env python3
"""
manga_processor.py
==================

C++のmain.cppとその関連ファイル群をPythonで再現した統合版
- ページ分割（2ページ → 1ページ）
- ページ分類（白ページ/黒ページ）
- フレーム（コマ）検出・抽出
- 吹き出し検出・抽出

Usage:
    python manga_processor.py <input_folder> <output_folder>
"""

import os
import sys
import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
from dataclasses import dataclass
import glob


@dataclass
class Point:
    """座標点を表すクラス"""
    x: int
    y: int


@dataclass
class Line:
    """直線を表すクラス（C++のLine classに相当）"""
    p1: Point
    p2: Point
    y2x: bool  # True: x = a*y + b (縦線), False: y = a*x + b (横線)
    a: float = 0.0
    b: float = 0.0
    
    def __post_init__(self):
        self._calc()
    
    def _calc(self):
        """傾きと切片を計算"""
        if self.y2x:
            # 縦線: x = a*y + b
            dy = self.p2.y - self.p1.y
            if dy == 0:
                self.a = 0.0
                self.b = float(min(self.p1.x, self.p2.x))
            else:
                self.a = (self.p2.x - self.p1.x) / dy
                self.b = self.p1.x - self.p1.y * self.a
        else:
            # 横線: y = a*x + b
            dx = self.p2.x - self.p1.x
            if dx == 0:
                self.a = 0.0
                self.b = float(min(self.p1.y, self.p2.y))
            else:
                self.a = (self.p2.y - self.p1.y) / dx
                self.b = self.p1.y - self.p1.x * self.a
    
    def judge_area(self, point: Point) -> int:
        """
        点が直線のどちら側にあるかを判定
        戻り値: 0=上/右側, 1=下/左側, 2=直線上
        """
        if self.y2x:
            x, y = point.y, point.x
        else:
            x, y = point.x, point.y
        
        value = self.a * x + self.b
        if y > value:
            return 0
        elif y < value:
            return 1
        else:
            return 2


@dataclass
class Points:
    """矩形の4つの角を表すクラス（C++のPoints classに相当）"""
    lt: Point  # left-top
    rt: Point  # right-top
    lb: Point  # left-bottom
    rb: Point  # right-bottom
    
    # 4辺の直線
    top_line: Optional[Line] = None
    bottom_line: Optional[Line] = None
    left_line: Optional[Line] = None
    right_line: Optional[Line] = None
    
    def renew_line(self):
        """4辺の直線を再計算（C++のrenew_line()に相当）"""
        self.top_line = Line(self.lt, self.rt, False)
        self.bottom_line = Line(self.lb, self.rb, False)
        self.left_line = Line(self.lb, self.lt, True)
        self.right_line = Line(self.rb, self.rt, True)
    
    def outside(self, p: Point) -> bool:
        """
        点が四角形の外部にあるかを判定（C++のoutside()に相当）
        戻り値: True=外部, False=内部または境界
        """
        if not all([self.top_line, self.bottom_line, self.left_line, self.right_line]):
            self.renew_line()
        
        return (self.top_line.judge_area(p) == 1 or
                self.right_line.judge_area(p) == 0 or
                self.bottom_line.judge_area(p) == 0 or
                self.left_line.judge_area(p) == 1)


@dataclass 
class Panel:
    """コマ（フレーム）情報"""
    image: np.ndarray  # RGBA画像
    bbox: Tuple[int, int, int, int]  # (x, y, w, h)
    corners: Points
    page_idx: int
    panel_idx: int


@dataclass
class Balloon:
    """吹き出し情報"""
    image: np.ndarray  # RGBA画像
    bbox: Tuple[int, int, int, int]  # (x, y, w, h)
    contour: np.ndarray
    center: Tuple[int, int]
    area: float
    circularity: float
    type: int  # 0:円形, 1:矩形, 2:ギザギザ
    bw_ratio: float
    panel_idx: int


class MangaProcessor:
    """マンガ処理メインクラス"""
    
    def __init__(self, input_folder: str, output_folder: str, evaluation_mode: bool = False):
        self.input_folder = Path(input_folder)
        self.output_folder = Path(output_folder)
        self.output_folder.mkdir(parents=True, exist_ok=True)
        self.evaluation_mode = evaluation_mode
        
        # 出力フォルダ構造
        self.panels_dir = self.output_folder / "panels"
        self.balloons_dir = self.output_folder / "balloons"
        self.panels_dir.mkdir(exist_ok=True)
        self.balloons_dir.mkdir(exist_ok=True)
        
        # 評価用出力ディレクトリ
        if self.evaluation_mode:
            self.eval_masks_dir = self.output_folder / "eval_masks"
            self.eval_instances_dir = self.output_folder / "eval_instances"
            self.eval_masks_dir.mkdir(exist_ok=True)
            self.eval_instances_dir.mkdir(exist_ok=True)
        
        # パラメータ設定
        self.speechballoon_max = 99  # 最大吹き出し数
        
    def get_image_paths(self, extension: str = "jpg") -> List[str]:
        """フォルダから画像パスを取得"""
        pattern = str(self.input_folder / f"*.{extension}")
        paths = glob.glob(pattern)
        paths.sort()
        return paths
    
    def page_cut(self, img: np.ndarray) -> List[np.ndarray]:
        """
        2ページ画像を1ページずつに分割
        C++の PageCut::pageCut() に相当
        """
        h, w = img.shape[:2]
        
        if w > h:  # 縦<横の場合:見開きだと判断し真ん中で切断
            # C++と同じ順序: 右ページ→左ページ
            cut_img_left = img[:, :w//2]      # 左半分（右ページ）
            cut_img_right = img[:, w//2:]     # 右半分（左ページ）
            return [cut_img_right, cut_img_left]  # 右ページを先に、左ページを後に
        else:  # 縦>横の場合:単一ページ画像だと判断しそのまま保存
            return [img]
    
    def get_page_type(self, page: np.ndarray) -> bool:
        """
        ページ分類：白ページ(False) / 黒ページ(True)
        C++の ClassificationPage::get_page_type() に相当
        """
        # C++と同じアルゴリズムを実装
        BLACK_LENGTH_TH = 5
        
        # フレーム存在領域を特定
        frame_exist_page = self.find_frame_area(page.copy())
        
        # 大津の手法で二値化
        _, frame_exist_page = cv2.threshold(frame_exist_page, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        
        h, w = frame_exist_page.shape
        
        # 上端チェック
        page_type = True
        for y in range(3, BLACK_LENGTH_TH):
            for x in range(w):
                if frame_exist_page[y, x] != 0:
                    page_type = False
                    break
            if not page_type:
                break
        if page_type:
            return True
        
        # 下端チェック
        page_type = True
        for y in range(h - 3, h - BLACK_LENGTH_TH, -1):
            for x in range(w):
                if frame_exist_page[y, x] != 0:
                    page_type = False
                    break
            if not page_type:
                break
        if page_type:
            return True
        
        # 右端チェック
        page_type = True
        for y in range(h):
            for x in range(w - 3, w - BLACK_LENGTH_TH, -1):
                if frame_exist_page[y, x] != 0:
                    page_type = False
                    break
            if not page_type:
                break
        if page_type:
            return True
        
        # 左端チェック
        page_type = True
        for y in range(h):
            for x in range(3, BLACK_LENGTH_TH):
                if frame_exist_page[y, x] != 0:
                    page_type = False
                    break
            if not page_type:
                break
        
        return page_type
    
    def find_frame_area(self, input_page_image: np.ndarray) -> np.ndarray:
        """
        フレーム存在領域特定（C++のfindFrameArea相当）
        """
        # グレースケール変換
        if len(input_page_image.shape) == 3:
            gray = cv2.cvtColor(input_page_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = input_page_image
        
        # ガウシアンフィルタ
        gaussian_img = cv2.GaussianBlur(gray, (3, 3), 0)
        
        # 階調反転二値画像の生成
        inverse_bin_img = cv2.bitwise_not(gaussian_img)
        _, inverse_bin_img = cv2.threshold(inverse_bin_img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        
        h, w = inverse_bin_img.shape
        
        # 左右の除去 - ヒストグラム生成
        histgram_lr = np.zeros(w, dtype=int)
        
        for y in range(h):
            for x in range(w):
                if x <= 2 or x >= w - 2 or y <= 2 or y >= h - 2:
                    continue
                if inverse_bin_img[y, x] > 0:
                    histgram_lr[x] += 1
        
        # 左右境界検出
        min_x_lr = 0
        max_x_lr = w - 1
        
        for x in range(w):
            if histgram_lr[x] > 0:
                min_x_lr = x
                break
        
        for x in range(w - 1, -1, -1):
            if histgram_lr[x] > 0:
                max_x_lr = x
                break
        
        # 誤差は両端に寄せる
        if min_x_lr < 6:
            min_x_lr = 0
        if max_x_lr > w - 6:
            max_x_lr = w
        
        cut_page_img_lr = gray[0:h, min_x_lr:max_x_lr]
        
        # 上下の除去 - ヒストグラム生成
        histgram_tb = np.zeros(h, dtype=int)
        
        for y in range(h):
            for x in range(w):
                if x <= 2 or x >= w - 2 or y <= 2 or y >= h - 2:
                    continue
                if inverse_bin_img[y, x] > 0:
                    histgram_tb[y] += 1
        
        # 上下境界検出
        min_y_tb = 0
        max_y_tb = h - 1
        
        for y in range(h):
            if histgram_tb[y] > 0:
                min_y_tb = y
                break
        
        for y in range(h - 1, -1, -1):
            if histgram_tb[y] > 0:
                max_y_tb = y
                break
        
        # 誤差は両端に寄せる
        if min_y_tb < 6:
            min_y_tb = 0
        if max_y_tb > cut_page_img_lr.shape[0] - 6:
            max_y_tb = cut_page_img_lr.shape[0]
        
        cut_page_img = cut_page_img_lr[min_y_tb:max_y_tb, :]
        
        return cut_page_img
    
    def frame_detect(self, page: np.ndarray) -> List[Panel]:
        """
        フレーム（コマ）検出・抽出
        C++の Framedetect::frame_detect() に相当
        """
        panels = []
        
        # 元画像を保存（パネル切り出しに使用）
        original_page = page.copy()
        
        # グレースケール変換（処理用）
        if len(page.shape) == 3:
            gray = cv2.cvtColor(page, cv2.COLOR_BGR2GRAY)
        else:
            gray = page.copy()  # 元画像を変更しないようにコピー
            
        # 吹き出し検出用の二値化
        _, bin_img = cv2.threshold(gray, 230, 255, cv2.THRESH_BINARY)
        bin_img = cv2.erode(bin_img, np.ones((3,3), np.uint8), iterations=1)
        bin_img = cv2.dilate(bin_img, np.ones((3,3), np.uint8), iterations=1)
        
        # 吹き出し輪郭検出
        contours, _ = cv2.findContours(bin_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        
        # 吹き出し除去（塗りつぶし）- 処理用画像のみ
        self.extract_speech_balloon(contours, gray)
        
        # ガウシアンフィルタ
        gaussian_img = cv2.GaussianBlur(gray, (3, 3), 0)
        
        # 二値化（反転）
        _, inverse_bin = cv2.threshold(gaussian_img, 210, 255, cv2.THRESH_BINARY_INV)
        
        # Cannyエッジ検出
        canny = cv2.Canny(gray, 120, 130, 3)
        
        # Hough直線検出
        lines = cv2.HoughLines(canny, 1, np.pi/180, 50)
        
        # 直線画像作成
        lines_img = np.zeros_like(gray)
        if lines is not None:
            for line in lines[:100]:  # 最大100本
                rho, theta = line[0]
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a * rho
                y0 = b * rho
                x1 = int(x0 - 2000 * b)
                y1 = int(y0 + 2000 * a)
                x2 = int(x0 + 2000 * b)
                y2 = int(y0 - 2000 * a)
                cv2.line(lines_img, (x1, y1), (x2, y2), 255, 1)
        
        # 論理積
        and_img = cv2.bitwise_and(inverse_bin, lines_img)
        
        # 輪郭検出
        contours, _ = cv2.findContours(and_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # バウンディングボックスで補完
        for cnt in contours:
            bbox = cv2.boundingRect(cnt)
            if self.judge_area_of_bounding_box(bbox, page.shape[0] * page.shape[1]):
                cv2.rectangle(and_img, bbox, 255, 3)
        
        # 最終的な補完画像
        complement_img = cv2.bitwise_and(and_img, inverse_bin)
        
        # 最終輪郭検出
        final_contours, _ = cv2.findContours(complement_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # パネル抽出（元画像から切り出し）
        for i, cnt in enumerate(final_contours):
            bbox = cv2.boundingRect(cnt)
            x, y, w, h = bbox
            
            if not self.judge_area_of_bounding_box(bbox, original_page.shape[0] * original_page.shape[1]):
                continue
                
            # 端に寄せる処理
            if x < 6: x = 0
            if y < 6: y = 0
            if x + w > original_page.shape[1] - 6: w = original_page.shape[1] - x
            if y + h > original_page.shape[0] - 6: h = original_page.shape[0] - y
            
            # C++版と同様の輪郭近似と四隅座標決定
            corners = self.define_panel_corners(cnt, bbox, original_page.shape)
            
            # 元画像からパネルを切り出し（吹き出しが塗りつぶされていない）
            panel_img = self.create_alpha_image(original_page, corners)
            
            # 切り出し
            panel_img = panel_img[y:y+h, x:x+w].copy()
            
            panel = Panel(
                image=panel_img,
                bbox=(x, y, w, h),
                corners=corners,
                page_idx=0,  # 後で設定
                panel_idx=i
            )
            panels.append(panel)
        
        return panels
    
    def extract_speech_balloon(self, contours: List[np.ndarray], img: np.ndarray):
        """
        吹き出し検出・塗りつぶし（フレーム検出の前処理用）
        C++の Framedetect::extractSpeechBalloon() に相当
        """
        img_area = img.shape[0] * img.shape[1]
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            peri = cv2.arcLength(cnt, True)
            
            if img_area * 0.008 <= area < img_area * 0.03:
                en = 4.0 * np.pi * area / (peri * peri + 1e-7)
                if en > 0.4:
                    cv2.drawContours(img, [cnt], -1, 0, -1)
    
    def define_panel_corners(self, contour: np.ndarray, bbox: Tuple[int, int, int, int], 
                           page_shape: Tuple[int, int]) -> Points:
        """
        C++版のdefinePanelCorners()に相当
        バウンディングボックスの角から最も近い輪郭点を四隅座標に決定
        """
        x, y, w, h = bbox
        
        # バウンディングボックスの四隅
        bb_lt = Point(x, y)      # left-top
        bb_rt = Point(x+w, y)    # right-top
        bb_lb = Point(x, y+h)    # left-bottom
        bb_rb = Point(x+w, y+h)  # right-bottom
        
        # 輪郭を多角形近似（C++のapproxPolyDP相当）
        epsilon = 6  # C++版と同じ値
        approx = cv2.approxPolyDP(contour, epsilon, True)
        approx_points = [Point(pt[0][0], pt[0][1]) for pt in approx]
        
        # 各角に最も近い輪郭点を探索
        def find_nearest_point(target_point: Point, candidates: List[Point]) -> Point:
            min_dist = float('inf')
            nearest_point = target_point
            
            for candidate in candidates:
                dist = ((target_point.x - candidate.x) ** 2 + 
                       (target_point.y - candidate.y) ** 2) ** 0.5
                if dist < min_dist:
                    min_dist = dist
                    nearest_point = candidate
            
            return nearest_point
        
        # 四隅の座標を決定
        definite_lt = find_nearest_point(bb_lt, approx_points)
        definite_rt = find_nearest_point(bb_rt, approx_points)
        definite_lb = find_nearest_point(bb_lb, approx_points)
        definite_rb = find_nearest_point(bb_rb, approx_points)
        
        # 端に寄せる処理（C++のalign2edge相当）
        th_edge = 6
        if definite_lt.x < th_edge: definite_lt.x = 0
        if definite_lt.y < th_edge: definite_lt.y = 0
        if definite_rt.x > page_shape[1] - th_edge: definite_rt.x = page_shape[1]
        if definite_rt.y < th_edge: definite_rt.y = 0
        if definite_lb.x < th_edge: definite_lb.x = 0
        if definite_lb.y > page_shape[0] - th_edge: definite_lb.y = page_shape[0]
        if definite_rb.x > page_shape[1] - th_edge: definite_rb.x = page_shape[1]
        if definite_rb.y > page_shape[0] - th_edge: definite_rb.y = page_shape[0]
        
        return Points(definite_lt, definite_rt, definite_lb, definite_rb)

    def create_alpha_image(self, src_page: np.ndarray, definite_panel_point: Points) -> np.ndarray:
        """
        C++のcreateAlphaImage()に相当
        四点座標を使用してピクセル単位でマスク処理
        """
        # RGBA変換
        if len(src_page.shape) == 3:
            rgba = cv2.cvtColor(src_page, cv2.COLOR_BGR2BGRA)
        else:
            rgba = cv2.cvtColor(src_page, cv2.COLOR_GRAY2BGRA)
        
        # 直線を更新
        definite_panel_point.renew_line()
        
        # ピクセル単位で四角形外部を透明化
        height, width = rgba.shape[:2]
        for y in range(height):
            for x in range(width):
                point = Point(x, y)
                if definite_panel_point.outside(point):
                    rgba[y, x, 3] = 0  # アルファチャンネルを0に
        
        return rgba

    def judge_area_of_bounding_box(self, bbox: Tuple[int, int, int, int], page_area: int) -> bool:
        """
        バウンディングボックス面積判定
        C++の Framedetect::judgeAreaOfBoundingBox() に相当
        """
        x, y, w, h = bbox
        return w * h >= 0.048 * page_area
    
    def speechballoon_detect(self, panel: np.ndarray) -> List[Balloon]:
        """
        吹き出し検出・抽出
        C++の Speechballoon::speechballoon_detect() に相当
        """
        balloons = []
        
        # グレースケール変換
        if len(panel.shape) == 4:  # BGRA
            gray = cv2.cvtColor(panel, cv2.COLOR_BGRA2GRAY)
        elif len(panel.shape) == 3:  # BGR
            gray = cv2.cvtColor(panel, cv2.COLOR_BGR2GRAY)
        else:
            gray = panel
        
        # 二値化
        _, bin_img = cv2.threshold(gray, 230, 255, cv2.THRESH_BINARY)
        
        # モルフォロジー処理
        kernel = np.ones((3, 3), np.uint8)
        bin_img = cv2.erode(bin_img, kernel, iterations=1)
        bin_img = cv2.dilate(bin_img, kernel, iterations=1)
        
        # 輪郭検出
        contours, _ = cv2.findContours(bin_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        
        panel_area = panel.shape[0] * panel.shape[1]
        
        for i, cnt in enumerate(contours):
            area = cv2.contourArea(cnt)
            peri = cv2.arcLength(cnt, True)
            
            if peri == 0:
                continue
                
            en = 4.0 * np.pi * area / (peri * peri)  # 円形度
            
            # C++と同じ面積と円形度フィルタ
            if not (panel_area * 0.01 <= area < panel_area * 0.9 and en > 0.4):
                continue
            
            # バウンディングボックス取得
            bbox = cv2.boundingRect(cnt)
            x, y, w, h = bbox
            
            center_x = x + w // 2
            center_y = y + h // 2
            
            # マスク作成（C++と同じ処理）
            mask = np.full(gray.shape, 255, dtype=np.uint8)  # 初期値255で初期化
            cv2.drawContours(mask, [cnt], -1, 0, 4)  # 輪郭を太さ4で0（黒）で描画
            cv2.drawContours(mask, [cnt], -1, 0, -1)  # 内部を0（黒）で塗りつぶし
            
            # C++のcopyTo処理を再現：マスクを使って背景をグレー（150）にする
            back_150 = np.full(gray.shape, 150, dtype=np.uint8)
            masked_img = gray.copy()
            # マスクが0（黒）の部分にback_150をコピー（C++のcopyTo処理）
            masked_img = np.where(mask == 0, back_150, gray)
            
            # 切り出し
            cropped_region = masked_img[y:y+h, x:x+w]
            
            # 白黒比率計算
            TH = 255 // 3  # 85
            B = np.sum((cropped_region < TH) & (cropped_region != 150))
            W = np.sum(cropped_region > (255 - TH))
            
            if W == 0 or not (0.01 < (B / W) < 0.7) or B < 10:
                continue
            
            # 形状判定
            max_rect = w * h * 0.95
            if area >= max_rect:
                set_type = 1  # 矩形
            elif en >= 0.7:
                set_type = 0  # 円形
            else:
                set_type = 2  # ギザギザ
            
            # コマサイズ判定
            if w * h >= panel_area * 0.9:
                continue
            
            # RGBA画像作成
            if len(panel.shape) == 3:
                rgba_panel = cv2.cvtColor(panel, cv2.COLOR_BGR2BGRA)
            elif len(panel.shape) == 2:
                # グレースケールの場合、RGBAに変換
                rgba_panel = cv2.cvtColor(panel, cv2.COLOR_GRAY2BGRA)
            else:
                rgba_panel = panel.copy()
            
            # 透明化
            alpha_mask = np.zeros(gray.shape, dtype=np.uint8)
            cv2.drawContours(alpha_mask, [cnt], -1, 255, -1)
            
            # RGBAチャンネルが存在する場合のみアルファを設定
            if len(rgba_panel.shape) == 3 and rgba_panel.shape[2] == 4:
                rgba_panel[:, :, 3] = alpha_mask
            
            # 切り出し
            balloon_img = rgba_panel[y:y+h, x:x+w].copy()
            
            balloon = Balloon(
                image=balloon_img,
                bbox=(x, y, w, h),
                contour=cnt,
                center=(center_x, center_y),
                area=area,
                circularity=en,
                type=set_type,
                bw_ratio=B / W if W > 0 else 0,
                panel_idx=0  # 後で設定
            )
            balloons.append(balloon)
        
        return balloons
    
    def create_evaluation_outputs(self, balloons: List[Balloon], image_shape: Tuple[int, int], 
                                  image_filename: str):
        """
        評価用の出力を生成
        
        Track A用: 二値マスク（全吹き出しを統合）
        Track B用: インスタンスラベルマップ（各吹き出しに異なるID）
        
        Parameters:
            balloons: 吹き出しリスト
            image_shape: 画像サイズ (height, width)
            image_filename: 出力ファイル名（拡張子なし）
        """
        if not self.evaluation_mode:
            return
        
        height, width = image_shape
        
        # Track A: 二値マスク（0=背景、255=吹き出し）
        binary_mask = np.zeros((height, width), dtype=np.uint8)
        
        # Track B: インスタンスラベルマップ（0=背景、1,2,3...=各吹き出し）
        instance_mask = np.zeros((height, width), dtype=np.uint8)
        
        for idx, balloon in enumerate(balloons, start=1):
            # Try to use contour if available; otherwise fallback to bbox
            contour = None
            try:
                if hasattr(balloon, 'contour') and balloon.contour is not None:
                    # Ensure contour is in the right shape for drawContours
                    contour = balloon.contour
                    if isinstance(contour, list):
                        # convert list of points to numpy array of shape (-1,1,2)
                        contour = np.array(contour, dtype=np.int32)
                    if contour.ndim == 2 and contour.shape[1] == 2:
                        contour = contour.reshape((-1, 1, 2))
                else:
                    contour = None
            except Exception:
                contour = None

            if contour is not None:
                # Track A: draw filled contour for binary mask
                cv2.drawContours(binary_mask, [contour], -1, 255, thickness=-1)
                # Track B: draw filled contour with instance id
                instance_id = min(idx, 254)
                cv2.drawContours(instance_mask, [contour], -1, int(instance_id), thickness=-1)
            else:
                # Fallback to bbox if contour is not available
                x, y, w, h = balloon.bbox
                cv2.rectangle(binary_mask, (x, y), (x + w, y + h), 255, -1)
                instance_id = min(idx, 254)
                cv2.rectangle(instance_mask, (x, y), (x + w, y + h), int(instance_id), -1)
        
        # 保存
        binary_path = self.eval_masks_dir / f"{image_filename}.png"
        instance_path = self.eval_instances_dir / f"{image_filename}.png"
        
        cv2.imwrite(str(binary_path), binary_mask)
        cv2.imwrite(str(instance_path), instance_mask)
    
    def remove_false_balloons(self, balloons: List[Balloon]) -> List[Balloon]:
        """
        誤検出除去処理
        C++の誤検出除去ロジックに相当
        """
        balloon_px_th = 5
        filtered_balloons = []
        
        for balloon in balloons:
            balloon_img = balloon.image
            h, w = balloon_img.shape[:2]
            
            # BGRAをグレースケールに変換
            gray_balloon = cv2.cvtColor(balloon_img, cv2.COLOR_BGRA2GRAY)
            
            # 二値化
            _, bin_balloon = cv2.threshold(gray_balloon, 150, 255, cv2.THRESH_BINARY)
            bin_balloon_bgra = cv2.cvtColor(bin_balloon, cv2.COLOR_GRAY2BGRA)
            bin_balloon_bgra[:, :, 3] = balloon_img[:, :, 3]
            
            # エッジマーキング
            marked_img = bin_balloon_bgra.copy()
            
            for y in range(h):
                for x in range(w):
                    p = marked_img[y, x]
                    
                    # 端から一定距離内または透明画素の周辺をマーク
                    if (x <= balloon_px_th or y <= balloon_px_th or 
                        x >= w - balloon_px_th or y >= h - balloon_px_th):
                        if p[3] != 0:
                            marked_img[y, x] = [255, 0, 0, p[3]]
                    else:
                        # 周辺透明画素チェック
                        is_edge = False
                        for th in range(1, balloon_px_th + 1):
                            neighbors = []
                            if y - th >= 0:
                                neighbors.append(marked_img[y - th, x])
                            if y + th < h:
                                neighbors.append(marked_img[y + th, x])
                            if x - th >= 0:
                                neighbors.append(marked_img[y, x - th])
                            if x + th < w:
                                neighbors.append(marked_img[y, x + th])
                            
                            for neighbor in neighbors:
                                if neighbor[3] == 0:
                                    is_edge = True
                                    break
                            if is_edge:
                                break
                        
                        if is_edge and p[3] != 0:
                            marked_img[y, x] = [255, 0, 0, p[3]]
            
            # 黒画素カウント
            edge_black_count = 0
            black_count = 0
            
            for y in range(h):
                for x in range(w):
                    p = marked_img[y, x]
                    
                    if p[3] != 0:
                        # 隣接赤マークチェック
                        is_adjacent_to_red = False
                        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                            ny, nx = y + dy, x + dx
                            if 0 <= ny < h and 0 <= nx < w:
                                neighbor = marked_img[ny, nx]
                                if (neighbor[0] == 255 and neighbor[1] == 0 and 
                                    neighbor[2] == 0):
                                    is_adjacent_to_red = True
                                    break
                        
                        # エッジ黒画素カウント
                        if (is_adjacent_to_red and 
                            not (p[0] == 255 and p[1] == 0 and p[2] == 0) and
                            p[0] == 0 and p[1] == 0 and p[2] == 0):
                            edge_black_count += 1
                        
                        # 全体黒画素カウント
                        if p[0] == 0 and p[1] == 0 and p[2] == 0:
                            black_count += 1
            
            # 判定
            if edge_black_count == 0 and black_count >= 100:
                filtered_balloons.append(balloon)
        
        return filtered_balloons
    
    def process_images(self):
        """メイン処理"""
        image_paths = self.get_image_paths()
        
        all_panels = []
        all_balloons = []
        
        print(f"Processing {len(image_paths)} images...")
        
        for i, image_path in enumerate(image_paths):
            print(f"Processing: {image_path}")
            
            # 画像読み込み（C++と同じグレースケール読み込み）
            img = cv2.imread(image_path, 0)  # 0でグレースケール読み込み
            if img is None:
                continue
            
            # ページ分割
            pages = self.page_cut(img)
            
            for j, page in enumerate(pages):
                # ページ分類
                is_black_page = self.get_page_type(page)
                if is_black_page:
                    print(f"  Skipping black page {i}_{j}")
                    continue
                
                print(f"  Processing page {i}_{j}")
                
                # フレーム検出
                panels = self.frame_detect(page)
                
                for k, panel in enumerate(panels):
                    panel.page_idx = j
                    panel.panel_idx = k
                    all_panels.append(panel)
                    
                    # パネル保存
                    panel_filename = f"{i:03d}_{j}_{k}.png"
                    cv2.imwrite(str(self.panels_dir / panel_filename), panel.image)
                    
                    # 吹き出し検出
                    balloons = self.speechballoon_detect(panel.image)
                    
                    if len(balloons) > self.speechballoon_max:
                        continue
                    
                    # 誤検出除去
                    filtered_balloons = self.remove_false_balloons(balloons)
                    
                    for l, balloon in enumerate(filtered_balloons):
                        balloon.panel_idx = len(all_panels) - 1
                        all_balloons.append(balloon)
                        
                        # 吹き出し保存
                        balloon_filename = f"{i:03d}_{j}_{k}_{l}.png"
                        cv2.imwrite(str(self.balloons_dir / balloon_filename), balloon.image)
        
        print(f"Processing complete!")
        print(f"Total panels: {len(all_panels)}")
        print(f"Total balloons: {len(all_balloons)}")
        
        return all_panels, all_balloons
    
    def process_single_image_for_evaluation(self, image_path: str) -> List[Balloon]:
        """
        評価用: 単一画像を処理して吹き出しを検出
        C++版と同様にページ分割→統合出力を行う
        
        Parameters:
            image_path: 画像ファイルパス
            
        Returns:
            検出された吹き出しのリスト
        """
        image_path = Path(image_path)
        image_filename = image_path.stem
        
        # 画像読み込み（C++と同じグレースケール読み込み）
        img = cv2.imread(str(image_path), 0)  # グレースケール
        if img is None:
            print(f"Error: Cannot load {image_path}")
            return []
        
        # 元画像のサイズ（統合マスク用）
        orig_h, orig_w = img.shape[:2]
        
        # ページ分割
        pages = self.page_cut(img)
        
        # 全ページの吹き出しを統合
        all_balloons = []
        
        for j, page in enumerate(pages):
            # ページ分類
            is_black_page = self.get_page_type(page)
            if is_black_page:
                continue
            
            # ページオフセット計算（C++版と同じロジック）
            page_offset_x = 0
            if orig_w > orig_h and len(pages) == 2:
                # 見開きページの場合
                # PageCutは [右ページ, 左ページ] の順で返す
                if j == 0:  # 右ページ
                    page_offset_x = orig_w // 2
                else:  # 左ページ
                    page_offset_x = 0
            
            # 吹き出し検出
            balloons = self.speechballoon_detect(page)
            
            # 誤検出除去
            filtered_balloons = self.remove_false_balloons(balloons)
            
            # bboxにページオフセットを適用
            for balloon in filtered_balloons:
                x, y, w, h = balloon.bbox
                balloon.bbox = (x + page_offset_x, y, w, h)
                
                # contourにもオフセットを適用
                if balloon.contour is not None:
                    balloon.contour = balloon.contour + np.array([page_offset_x, 0])
            
            all_balloons.extend(filtered_balloons)
        
        # 評価用出力を生成（統合された全吹き出しで1枚のマスク）
        if self.evaluation_mode:
            self.create_evaluation_outputs(
                all_balloons, 
                (orig_h, orig_w),  # 元画像サイズ
                image_filename
            )
        
        return all_balloons
    
    def process_evaluation_dataset(self, image_dir: Path):
        """
        評価データセットの全画像を処理
        
        Parameters:
            image_dir: 画像ディレクトリ（evaluation_dataset/sampled_images等）
        """
        image_dir = Path(image_dir)
        
        # 画像ファイル一覧取得
        image_files = []
        for ext in ['*.jpg', '*.JPG', '*.png', '*.PNG']:
            image_files.extend(sorted(image_dir.glob(ext)))
        
        if len(image_files) == 0:
            print(f"No images found in {image_dir}")
            return
        
        print(f"Processing {len(image_files)} images for evaluation...")
        
        total_balloons = 0
        
        for idx, image_path in enumerate(image_files):
            balloons = self.process_single_image_for_evaluation(str(image_path))
            total_balloons += len(balloons)
            
            if (idx + 1) % 10 == 0:
                print(f"  Processed {idx + 1}/{len(image_files)} images...")
        
        print(f"\nEvaluation processing complete!")
        print(f"Total images: {len(image_files)}")
        print(f"Total balloons detected: {total_balloons}")
        print(f"Average balloons per image: {total_balloons / len(image_files):.2f}")
        
        if self.evaluation_mode:
            print(f"\nEvaluation outputs:")
            print(f"  Binary masks (Track A): {self.eval_masks_dir}")
            print(f"  Instance masks (Track B): {self.eval_instances_dir}")


def main():
    if len(sys.argv) < 3:
        print("Usage:")
        print("  python manga_processor.py <input_folder> <output_folder> [--eval]")
        print("\nOptions:")
        print("  --eval    Enable evaluation mode (generates Track A/B outputs)")
        print("\nExamples:")
        print("  # Normal mode")
        print("  python manga_processor.py ./images ./output")
        print("\n  # Evaluation mode")
        print("  python manga_processor.py ./evaluation_dataset/sampled_images ./predictions --eval")
        sys.exit(1)
    
    input_folder = sys.argv[1]
    output_folder = sys.argv[2]
    evaluation_mode = '--eval' in sys.argv
    
    if not os.path.exists(input_folder):
        print(f"Error: Input folder '{input_folder}' does not exist.")
        sys.exit(1)
    
    processor = MangaProcessor(input_folder, output_folder, evaluation_mode=evaluation_mode)
    
    if evaluation_mode:
        # 評価モード: 単一ページ画像を処理
        processor.process_evaluation_dataset(input_folder)
    else:
        # 通常モード: 見開きページ分割等を行う
        panels, balloons = processor.process_images()


if __name__ == "__main__":
    main()