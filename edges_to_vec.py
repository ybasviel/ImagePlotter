import random
import math
import numpy as np


def edges2polylines(edges, th_n=6, th_c=None):
    """
    cv2.Canny()から返されるエッジ検出結果からエッジ点を読みより折れ線に変換する

    Parameters
    ----------
    edges : numpy.ndarray
        cv2.Canny()から返されるエッジ検出結果
    th_n : int or float
        近傍とみなす点の距離の閾値
    th_c : int or float
        許容する目的関数の最小値の閾値

    Returns
    -------
    polylines : list of list of numpy.ndarray
        折れ線のリスト[l1, l2, ... ]
    """
    edges_tmp = edges.copy()
    th_c = th_c or th_n ** 2
    edgepoints = get_edgepoints(edges_tmp)  # エッジ点の集合 P
    init_n_points = len(edgepoints)
    polylines = []  # 折れ線 l1, l2, ... を格納するリスト
    count = 0
    while count < init_n_points:
        edgepoints = get_edgepoints(edges_tmp)
        n_points = len(edgepoints)
        init_idx = random.randrange(n_points)

        # polyline = []
        polyline = np.empty((0, 2), dtype=np.float32)   # 折れ線 li
        #polyline.append(convert_imshow2plot(edgepoints[init_idx]))
        a = convert_imshow2plot(edgepoints[init_idx])
        polyline = np.append(polyline, a.reshape(1,2), axis=0)
        
        delete_edges(edges_tmp, edgepoints[init_idx])
        count += 1
        could_reverse = True
        while True:
            # 折れ線の最後尾の近傍にあるエッジ点を取得する
            nearest = search_neighbor_points(edges_tmp, polyline[-1, :], th_n)
            # 線の端に近い順に並び替え
            nearest = sorted(nearest, key=lambda p: distance(p, polyline[-1, :]))
            # 近傍にエッジ点がなかった時の処理
            if not nearest:
                if could_reverse:
                    np.flip(polyline, 0)
                    could_reverse = False
                    continue
                else:
                    # 近くに点もなく，既に反転済みのときは線の作成を終える
                    break
            if np.shape(polyline)[0] <= 2:
                next_point = nearest[0]
            else:
                # line[-2]とline[-3]の中点からline[-1]へ向かうベクトル
                d = (polyline[-1, :] - polyline[-2, :]) + 0.47 * (polyline[-2, :] - polyline[-3, :])
                d = d / np.linalg.norm(d)  # 正規化
                nearest = sorted(nearest, key=lambda p: cost_function(p, polyline[-1, :], d))
                # 目的関数の最小値が閾値を超えていた時の処理
                if cost_function(nearest[0], polyline[-1, :], d) > th_c:
                    if could_reverse:
                        np.flip(polyline, 0)
                        could_reverse = False
                        continue
                    else:
                        break
                next_point = nearest[0]
            
            # polyline.append(next_point)
            polyline = np.append(polyline, next_point.reshape(1, 2), axis=0)
            
            delete_edges(edges_tmp, convert_plot2imshow(next_point))
            count += 1
            
        polylines.append(polyline)
    return polylines


def cost_function(p, line_end, d):
    """
    目的関数

    Parameters
    ----------
    p : numpy.ndarray
        エッジ点
    line_end : numpy.ndarray
        折れ線の末端のエッジ点
    d : numpy.ndarray
        曲がり具合を計算するために用いる単位ベクトル

    Returns
    -------
    res : float
        目的関数の評価値
    """
    r = p - line_end
    r_norm = math.sqrt(np.dot(r, r))
    res = r_norm ** 2 - 0.99 * np.dot(r, d)
    return res


def search_neighbor_points(edges, a, th_c):
    """
    edgesの中から点aからth_c以内の距離にある点を抽出する

    Parameters
    ----------
    edges : numpy.ndarray
        cv2.Canny()から返されるエッジ検出結果
    a  : numpy.ndarray
        点
    th_c : int or float
        近傍と見なす距離の閾値

    Returns
    -------
    neighbor_points : list of numpy.ndarray
        aの近傍にある点のリスト
    """
    neighbor_points = []
    pl = points_inside_circle(a, th_c)
    for p in pl:
        p_img = convert_plot2imshow(p)
        if p_img[0] < 0 or p_img[1] < 0:
            continue
        try:
            if edges[p_img[0], p_img[1]] == 255:
                neighbor_points.append(p)
        except IndexError:
            continue
    return neighbor_points


def distance(a, b):
    """
    2点間のユークリッド距離を求める(2次元)

    Parameters
    ----------
    a : array-like
        点1
    b : array-like
        点2

    Returns
    -------
    d : float
        二点間のユークリッド距離
    """
    d = math.sqrt((a[0] - b[0]) ** 2 +
                  (a[1] - b[1]) ** 2)
    return d


def get_edgepoints(edges):
    """
    cv2.Canny()から返されるエッジ検出結果からエッジ点の位置を取得する

    Parameters
    ----------
    edges : numpy.ndarray
        cv2.Canny()から返されるエッジ検出結果

    Returns
    -------
    edgepoints : list of numpy.ndarray
        エッジ点の位置のリスト
        [[x1,y1], [x2,y2], ... ]
    """
    il, jl = np.where(edges == 255)
    edgepoints = [np.array([i, j]) for i, j in zip(il, jl)]  # エッジ点の集合 P
    return edgepoints


def points_inside_circle(center, r):
    """
    centerを中心とする半径rの円の内部または円周上にある格子点のリストを返す

    Parameters
    ----------
    center : numpy.ndarray
        円の中心
    r : int or float
        円の半径

    Returns
    -------
    pl : list of numpy.ndarray
        円の内部または円周上にある格子点のリスト
    """
    pl = []
    ri = math.floor(r)
    xl = np.arange(-ri, ri + 1)
    for x in xl:
        y = math.sqrt(r ** 2 - x ** 2)
        yi = math.floor(y)
        yl = np.arange(-yi, yi + 1)
        pl.extend([np.array([x, y]) + center for y in yl])
    return pl


def delete_edges(edges, p):
    """
    edgesの指定された点を0にする

    Parameters
    ----------
    edges : numpy.ndarray
        cv2.Canny()から返されるエッジ検出結果
    p : numpy.ndarray
        0にする点(インデックス)
    """
    edges[p[0], p[1]] = 0


def convert_imshow2plot(p):
    """
    点pの座標系をimshowの座標系からplotの座標系へ変換する

    Parameters
    ----------
    p : numpy.ndarray
        座標系を変換する点

    Returns
    -------
    tmp : numpy.ndarray
        変換済みの点p
    """
    tmp = p.copy()
    tmp[0] = p[1]
    tmp[1] = - p[0]
    return np.array([p[1], -p[0]])
    return tmp


def convert_plot2imshow(p):
    """
    点pの座標系をplotの座標系からimshowの座標系へ変換する

    Parameters
    ----------
    p : numpy.ndarray
        座標系を変換する点

    Returns
    -------
    tmp : numpy.ndarray
        変換済みの点p
    """
    tmp = p.copy()
    tmp[0] = - p[1]
    tmp[1] = p[0]
    tmp = tmp.astype(np.int64)
    return tmp

