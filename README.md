
“u + iv” で 2 成分を 1 本化して扱う発想

アプローチ	何を 1 本にまとめるか	実装の要点	推奨シーン
① 複素表現(u,v) → w = u + i v	**振幅＝ 	w	，位相＝arg w** に分解しやすく、回転は位相の平行移動に対応
② POD/SPOD を複素で	スナップショット行列 X = [w₁ … wₙ]（各列＝u+iv をフラット）	- 共分散 C = XᴴX / n を固有分解→複素 POD モード- 周波数別に分離したい場合は SPOD を使う ￼	小サンプル (≲50) でも OK、かつモード解釈を重視
③ Helmholtz 分解	div w（スカラー）と curl w（スカラー）を派生量として 2ch へ	- div w = ∂u/∂x + ∂v/∂y, ω = ∂v/∂x – ∂u/∂y- これを 1 枚の 2ch 画像にして CNN・POD 等	流れ場など 発散 / 渦度 が物理的意味をもつ場合 ￼ ￼
④ E(2)‐エッジ巻き込み (等方 CNN/GNN)	ベクトル場まるごとを 1 テンソルで入力し、ネットワークが回転・反転に自動適合	- PyTorch e2cnn ライブラリ ￼- 転移防止用に FieldType=R2 表現を指定	データ拡張が難しい・大きめサンプル (≫100) がある場合
⑤ 複素ラジアル基底 (RBF, Zernike)	w を複素 RBFや複素 Zernike係数に直結	- 実部・虚部を連結して 2k 次元→MLP/Ridge- 位相不変量 `	a_k
⑥ スカラー化 (振幅＋角度 PE)	`	w	とsin θ, cos θ (θ=arg w)` を3 チャンネルに展開
⑦ 構造テンソル (2×2)	T = [[u², uv],[uv, v²]] → 6 スカラー	- 各点で方向無関係の二次モーメント- 面平均や低次モーメントを特徴に	パターン方向を無視して強度・異方性だけ欲しい


⸻

コア実装例（複素 POD＋リッジ）

W = []                       # list of (Nx*Ny,) complex snapshots
for u,v in dataset:          # u,v : (Nx,Ny)
    W.append((u+1j*v).ravel())
X = np.column_stack(W)       # shape (Npix, Nsamp)

#   1) 複素 POD
C  = X.conj().T @ X / X.shape[1]
eigval, eigvec = np.linalg.eigh(C)
idx = eigval.argsort()[::-1]
Phi = (X @ eigvec[:,idx[:r]])    # モード (Npix,r)

#   2) 複素係数 → 実ベクトル化
A   = np.linalg.lstsq(Phi, X, rcond=None)[0]  # shape (r,Nsamp)
A_reim = np.vstack([A.real, A.imag]).T        # (Nsamp, 2r)

#   3) 10 ゾーン入力 → リッジで係数回帰
from sklearn.linear_model import RidgeCV
ridge = RidgeCV(alphas=[1e-2,1e-1,1])
ridge.fit(inputs_10x, A_reim)                 # 10 → 2r

#   4) 推論 → 再構成
a_hat = ridge.predict(new_input[None,:]).T    # (2r,1)
a_c   = a_hat[:r] + 1j*a_hat[r:]              # 複素
w_rec = Phi @ a_c                             # vector field flat
u_rec = w_rec.real.reshape(Nx,Ny)
v_rec = w_rec.imag.reshape(Nx,Ny)

	•	学習パラメータ ~20–40 で収まり、30 サンプルでも過学習しにくい
	•	複素モードなので回転が位相だけに乗り、モデルが扱いやすい

⸻

手法の選び方の目安

データ量	回転や反転を厳密に扱いたい	推奨ルート
≤ 50	★★★（解釈重視）	複素 POD/SPOD → リッジ/GPR  ￼ ￼
50–200	★★	複素基底 (Zernike/RBF) 係数＋ランダムフォレスト
≫ 200	★★★（高精度）	E(2)‐Equivariant CNN/GNN  ￼ ￼
任意	★	振幅+方向 PE (


⸻

まとめ
	•	**「u と v を別々に扱わない」**ためには
	1.	複素数 w = u + iv や (mag, dir) に写像して1 系列に束ねる
	2.	低次モード抽出（POD/Zernike）で数十次元以下へ圧縮
	3.	小データなら 線形回帰か GPR、大データなら E(2)‐Equivariant NN
	•	物理量として「回転」が鍵なら複素表現＋位相が最も素直
	•	回転不変が欲しい場合は |w| や構造テンソル、
回転依存を学習させたい場合は sinθ,cosθ PE や E(2)‐CNN が有効

これで x,y 成分を 1 つの特徴表現に統合しつつ、
データ量や精度要求に合わせて単一モデルで扱う設計が可能です。  ￼ ￼ ￼ ￼ ￼

公開ベースで探せる「円盤状（ウェハー）データ」の現状

種類	主な公開ソース	内容	ベクトル場？	備考
ウェハー欠陥マップ	WM-811K (MIT, Kaggle)  ￼WaferMap / MixedWM38 (GitHub)  ￼	ダイ格子上の 2値 or 多値クラス画像（defect / no-defect 等）	×（スカラー画像）	81 万枚超。深層学習のベンチマークとして定番
ウェハー高さ／平坦度測定	NIST, MDPI 等の論文付録（例: IR干渉計サンプル） ￼	z(x,y) の高さマップ（μm〜nm）	△（スカラー→∇z を取れば疑似ベクトルに）	データは図表のみ or 1〜2 枚の CSV で規模小
オーバレイ／ひずみベクトル	学会論文の図面例 (例: Ku et al., 2013)  ￼	リソグラフィ位置ずれ ベクトル矢印	▲（PDF 図面、元CSV非公開）	多くは製造機密で 生データ未公開
合成 PIV・流体円盤スライス	JHTDB 2D スライス API	u,v 乱流場（円盤ではなく矩形）	○（円形切り抜き可）	※ウェハー物理ではないが円盤マスク可

結論:
	•	“x–y ベクトルをそのまま公開” したウェハー実測データセットは現状ほぼ存在しません。
	•	オーバレイ誤差やウェハー変位ベクトルは各半導体メーカーの機密に直結し、学術公開されにくい。
	•	一方で 欠陥マップ（スカラー）や平坦度の高さマップ は比較的公開例があり、これらを勾配や位相変換して疑似ベクトル場を生成することは技術的には可能です。

⸻

回避策 ①　高さマップ → 勾配ベクトル場
	1.	データ入手
	•	論文付録の CSV 例（6 inch, 300 mm wafer 高さ）やメーカーカタログのサンプル測定値を取得。
	2.	処理

import numpy as np
dzdx, dzdy = np.gradient(z_height, dx, dy)  # ベクトル (−∂z/∂x, −∂z/∂y)


	3.	出力
	•	勾配ベクトルを u = −∂z/∂x, v = −∂z/∂y と解釈すれば「反り方向ベクトル分布」。

→ これで “曲がり／反りベクトル” という物理量が得られ、Zernike展開・PODなどそのまま適用可能。

⸻

回避策 ②　公開欠陥マップ → 合成変位ベクトル

目的: 欠陥クラスタ位置を “力点” とみなし、ウェハー平面への擬似応力・変位場を可視化して ML ベンチを作る。
	1.	WM-811K の 512×512 バイナリマップを読み込み。
	2.	欠陥ピクセルを 点荷重 f(x,y) に見立て、
	•	2D ポアソン方程式：∇²Φ = −f → Φ を解き displacement ≈ ∇Φ
	3.	Φ の x,y 勾配を u,v として新しいベクトル場データセットを構築。

⸻

回避策 ③　シミュレーション／パラメトリック生成

製造工程を簡略モデル化し、Zernike 係数 ± ランダム局所ひずみ で合成

ステップ	例
1. 基底	Zernike n≤6：Tilt, Defocus, Astig, Coma, Trefoil …
2. ランダム局所異常	ガウシアン歪み `u(x,y)=A·exp(−
3. ノイズ	1/f 型スペクトル or 白色ノイズ β%
4. 出力	u,v ベクトル場 & 10 ゾーン平均値 ⇒ 教師データ

メリット – ラベル（真の Zernike 係数や欠陥パラメータ）が自明なので、教師ありでも物理解釈付きでも回せる。

⸻

公共データが増えそうな動き
	•	AI 製造研究用プラットフォーム：IME-Singapore が 2024 年発表した「Virtual Fab」では、一部 8-inch Si ウェハーの薄膜‐応力シミュレーション結果の共有を計画中（実データはまだ非公開）。
	•	学会附属チャレンジ：SPIE Advanced Lithography では 2025 年から Overlay Vector Prediction Challenge を検討中との報告あり（ワンオフ公開の可能性）。

⸻

まとめ
	1.	現時点で「円盤上の実測ベクトル場」を丸ごと公開するデータセットは見当たらない。代替として
	•	高さマップ → 勾配ベクトル
	•	欠陥マップ → ポアソン解釈の仮想変位
	•	Zernike 合成 → 疑似データ
	2.	もし実ベクトルデータが必須なら
	•	装置メーカや研究機関 (NIST, imec, CEA-Leti) の共同研究枠で入手交渉
	•	産学連携のベンチマーク企画（学会 Challenge） に参加してデータ提供を受ける
	3.	研究・PoC 段階なら 合成／勾配変換データ でも、特徴量設計・モデル選定・不確かさ評価の検証は十分可能。



結論 ― 現時点で「ウェハー上のオーバレイ／ひずみベクトル（u-v フィールド）をそのままダウンロードできる“公開 API 付きデータセット”は存在しません。

オーバレイ・IPD（In-Plane Distortion）・ひずみベクトルは パターン位置ずれ＝量産歩留りに直結するファブの機微情報 であり、論文図中のベクトルプロットはあっても 生データは非公開 が実情です。以下に 「代わりに入手可能なルート」 と 「商用ツールのエクスポート API／URL」 を整理します。

利用可能性	データ源 / API	要点	URL／リファレンス
△ 図のみ	学術論文の付録図（例：Ku et al., Optics Express 2013）	TIS 補正後オーバレイベクトルを図で公開。著者に依頼すれば CSV を共有してくれる場合も	￼
○ 疑似データ生成	WM-811K 欠陥マップ (811 k 枚) などを勾配→疑似ベクトル化	スカラー欠陥画像をポアソン解釈して変位ベクトルを合成	￼
○ 参考用ベクトル場	JHTDB (乱流 DNS) REST API	任意 2D スライス u,v を取得可。円盤マスクを掛ければ「ベクトル場×円形領域」実験が可能	￼ ￼
◎ 製造現場データ (有償)	KLA PWG™ / WaferSight™ シリーズ	Patterned Wafer Geometry (PWG) ツールが .csv で IPD / overlay ベクトルを書き出し（装置ライセンス契約が必要）	￼ ￼
	ASML YieldStar, Nova 光散乱メトロロジ	Overlay Δx,Δy をフィールド単位で API エクスポート（SECS/GEM, XML）	Nova 製品概要  ￼
◎ コラボ／受託	imec ・ CEA-Leti ・ NIST 共同研究枠	テストウェハーの PWG/overlay raw を NDA 付きで提供する事例あり	研究枠の公募情報を随時確認


⸻

1. どうしても「実ベクトル」を使いたい場合の選択肢

ルート	具体的な入手方法
A. 製造装置ログ	ファブ内で KLA PWG / ASML YieldStar が生成する *.csv (2000〜3000 site/wafer) をそのまま利用。社外持ち出しは NDA が必須。
B. 共同研究	研究機関経由で Engineered Stress Monitor (ESM) テストウェハーの PWG + Overlay ファイルを入手する例がある (例: SPIE 2024 論文).
C. 合成データ	既知 Zernike 係数 + ランダム局所歪みでベクトル場を合成し、10 ゾーン平均を教師信号にする――モデル検証に十分使える。


⸻

2. JHTDB を「円盤＋ベクトル場」サロゲートとして使う手順

import pyJHTDB, numpy as np

# ① 認証トークン取得 (https://turbulence.pha.jhu.edu/) 
token = 'your_token'

# ② 乱流データベースから 2D スライスを取得
db   = pyJHTDB.dbclient()
db.init(token=token)
field = db.getData("isotropic1024coarse", (t, z, y0:y1, x0:x1), data_type="velocity")

# ③ 円盤マスクを掛け、u,v をウェハー座標にリスケール
Nx, Ny = field.shape[2:]
y, x   = np.ogrid[-Ny/2:Ny/2, -Nx/2:Nx/2]
mask   = x**2 + y**2 <= (Nx/2)**2
u, v   = field[0,0] * mask, field[0,1] * mask

REST エンドポイント：https://turbulence.pha.jhu.edu/   ￼

⸻

3. まとめ
	•	公開 API でそのままダウンロードできる“ウェハーのオーバレイ／ひずみベクトル”データセットは現状なし
	•	生データは各社メトロロジ装置 (KLA PWG™ など) の 装置内エクスポート機能 に限られる。
	•	研究目的で手早く試すなら
	1.	JHTDB などオープンなベクトル場を円盤マスクで代用
	2.	WM-811K 欠陥マップや高度マップを勾配変換して疑似ベクトル生成
	•	量産レベルの実測値が必要な場合は
	•	装置メーカ（KLA, Nova, ASML）へ評価ライセンスを依頼
	•	imec / NIST の共同研究プログラムでテストウェハーの PWG / Overlay ファイルを取得
	•	SPIE・SEMICON 併設の overlay challenge 公開データをウォッチ（2025 企画検討中）

こうした事情から、まずは 疑似データ or JHTDB でモデルの枠組みを検証し、
その後 NDA 下で実測 CSV を流し込むフェーズに進むのが現実的です。