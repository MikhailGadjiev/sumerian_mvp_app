import json
import os
from pathlib import Path

from sklearn.cluster import DBSCAN
import numpy as np
import re

import re

REPO_ROOT = Path(__file__).resolve().parents[2]
unicode_to_charname_json = REPO_ROOT / "dataset/MaiCuBeDa/utils/utils/unicode_to_signnames.json"
with open(unicode_to_charname_json,'r',encoding='utf-8') as f:
    unicode_to_charname = json.load(f)
    
ACCENTED_CHARS = ["Á","´","Û","Ù","Ü","È","É","Ê","Ë","Ì","Í","Î","Ï","Ò","Ó","Ô","Ö","à","á","â","ä","è","é","ê","ë","ì","í","î","ï","ò","ó","ô","ö","ù","ú","û","ü","Ā","ā","Ē","ē","Ī","ī","Ō","ō","Ū","ū"]
charname_to_unicode ={}
for unicode, charname in unicode_to_charname.items():
    charname = charname["signName"].split(" (")[0]
    # remove any non-English accent marked characters
    for accented_char in ACCENTED_CHARS + [c.lower() for c in ACCENTED_CHARS]:
        charname = charname.replace(accented_char, "")
    charname = re.sub(r"\(.*?\)","",charname).strip()
    charname_to_unicode[charname] = unicode


def CER(s1, s2):
    """
    Calculate the Character Error Rate (CER) between two strings.
    CER = EditDistance(s1, s2) / len(s2)
    
    Args:
        s1 (str): predicted string
        s2 (str): ground truth string
        
    Returns:
        float: CER value
    """
    # Initialize DP matrix
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    # Base cases
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    # Compute edit distance
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if s1[i - 1] == s2[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,      # deletion
                dp[i][j - 1] + 1,      # insertion
                dp[i - 1][j - 1] + cost  # substitution
            )

    edit_distance = dp[m][n]
    
    return edit_distance / n if n > 0 else float('inf')

def filter_bboxes(bboxes, texts):

    # preprocess: if a box is 80% inside another box, remove it
    filtered_bboxes = []
    filtered_texts = []
    for i, (box_i, text_i) in enumerate(zip(bboxes, texts)):
        x1_i, y1_i, x2_i, y2_i = box_i
        area_i = (x2_i - x1_i) * (y2_i - y1_i)
        inside = False
        for j, box_j in enumerate(bboxes):
            if i == j:
                continue
            x1_j, y1_j, x2_j, y2_j = box_j
            inter_area = max(0, min(x2_i, x2_j) - max(x1_i, x1_j)) * max(0, min(y2_i, y2_j) - max(y1_i, y1_j))
            if inter_area / area_i >= 0.8:
                inside = True
                break
        if not inside:
            filtered_bboxes.append(box_i)
            filtered_texts.append(text_i)
            
    return filtered_bboxes, filtered_texts

def reading_order(bboxes, texts, alpha=0.8):
    """
    Sort bboxes in reading order using dynamic eps for DBSCAN.
    alpha: multiplier for line separation threshold.
    """

    
    assert len(bboxes) == len(texts)
    
    # Compute centers and heights
    centers = []
    heights = []
    filtered_bboxes, texts = filter_bboxes(bboxes, texts)
    for (x1, y1, x2, y2) in filtered_bboxes:
        centers.append([(x1 + x2) / 2, (y1 + y2) / 2])
        heights.append(y2 - y1)
    centers = np.array(centers)
    heights = np.array(heights)

    # Compute dynamic eps
    median_h = np.median(heights)
    eps = median_h * alpha

    # Cluster by vertical position (y-center)
    ys = centers[:, 1].reshape(-1, 1)
    clustering = DBSCAN(eps=eps, min_samples=1).fit(ys)
    line_ids = clustering.labels_

    # Sort line clusters top-to-bottom
    line_order = sorted(
        set(line_ids),
        key=lambda lid: np.mean(ys[line_ids == lid])
    )

    # Sort inside each line left-to-right
    ordered_text = []
    for lid in line_order:
        idxs = np.where(line_ids == lid)[0]
        idxs = sorted(idxs, key=lambda i: centers[i][0])
        ordered_text.extend(texts[i] for i in idxs)

    return "".join(ordered_text)
def charnames_to_unicode_string(charname):
    charname = charname.replace("__",", ")
    charname = charname.replace("_"," ")
    charname = re.sub(r"\(.*?\)","",charname).strip()
    if "+" not in charname:
        charnames = [charname]
    else:
        charnames = charname.strip("+").split("+")
    unicode_string = ""
    if len(charnames)>1:
        print(f"Info: charname '{charname}' split into multiple charnames: {charnames}")
    for cname in charnames: 
        cname=cname.strip()
        if cname in charname_to_unicode:
            unicode_ = charname_to_unicode[cname]
            unicode_char = chr(int(unicode_.replace("U+",""),16))
            unicode_string += unicode_char
        else:
            print(f"Warning: charname '{cname}' not found in charname_to_unicode mapping.")
            unicode_string += "?"
    return unicode_string