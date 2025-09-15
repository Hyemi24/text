import io
import os
import streamlit as st
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# ===== 한글 폰트 설정 =====
FONT_PATH = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"
matplotlib.rcParams["font.family"] = "NanumGothic"
matplotlib.rcParams["axes.unicode_minus"] = False

st.set_page_config(page_title="국어 텍스트 데이터 시각화", page_icon="🧠")

# ===== 토크나이저: Kiwi (Java 불필요, 설치 빠름) =====
try:
    from kiwipiepy import Kiwi
    kiwi = Kiwi()
    def tokenize_nouns(text: str):
        # 명사/고유명사만 추출
        return [w.form for w in kiwi.tokenize(text or "") if w.tag in ("NNG", "NNP")]
except Exception as e:
    kiwi = None
    st.warning("Kiwi 초기화 실패. 간단 토크나이저로 대체합니다.")
    import re
    hangul = re.compile(r"[가-힣]{2,}")
    def tokenize_nouns(text: str):
        return hangul.findall(text or "")

# ===== 공공 도메인 텍스트(원문) 샘플 =====
FULL_TEXT_SAMPLES = {
    "직접 입력": "",
    "시 — 김소월 〈진달래꽃〉 (원문)": """나 보기가 역겨워 가실 때에는
말없이 고이 보내 드리우리다
영변에 약산 진달래꽃
아름 따다 가실 길에 뿌리우리다

가시는 걸음걸음 놓인 그 꽃을
사뿐히 즈려 밟고 가시옵소서
나 보기가 역겨워 가실 때에는
죽어도 아니 눈물 흘리우리다""",
    "시 — 윤동주 〈서시〉 (원문)": """죽는 날까지 하늘을 우러러
한 점 부끄럼이 없기를,
잎새에 이는 바람에도
나는 괴로워했다.

별을 노래하는 마음으로
모든 죽어가는 것을 사랑해야지
그리고 나에게 주어진 길을
걸어가야겠다.

오늘 밤에도 별이 바람에 스치운다.""",
    # 아래 소설 3편은 실제 수업용이라면 위키문헌 전문을 복사해서 붙여 넣으세요.
    "소설 — 김유정 〈동백꽃〉 (원문)": """(여기에 위키문헌 전문을 붙여넣으세요)""",
    "소설 — 현진건 〈운수 좋은 날〉 (원문)": """(여기에 위키문헌 전문을 붙여넣으세요)""",
    "소설 — 이효석 〈메밀꽃 필 무렵〉 (원문)": """(여기에 위키문헌 전문을 붙여넣으세요)""",

    # 기사: 제목만 제공(본문은 저작권상 직접 입력)
    "기사 — (오늘) AI·교육 베스트 #1": "※ 기사 본문은 저작권 때문에 제공할 수 없습니다.\n기사 본문을 아래 입력란에 붙여넣고 분석해 보세요.",
    "기사 — (오늘) AI·교육 베스트 #2": "※ 기사 본문은 저작권 때문에 제공할 수 없습니다.\n기사 본문을 아래 입력란에 붙여넣고 분석해 보세요.",
    "기사 — (오늘) AI·교육 베스트 #3": "※ 기사 본문은 저작권 때문에 제공할 수 없습니다.\n기사 본문을 아래 입력란에 붙여넣고 분석해 보세요.",
    "기사 — (오늘) AI·교육 베스트 #4": "※ 기사 본문은 저작권 때문에 제공할 수 없습니다.\n기사 본문을 아래 입력란에 붙여넣고 분석해 보세요.",
    "기사 — (오늘) AI·교육 베스트 #5": "※ 기사 본문은 저작권 때문에 제공할 수 없습니다.\n기사 본문을 아래 입력란에 붙여넣고 분석해 보세요.",
}

# ===== 세션 상태 =====
if "input_text" not in st.session_state:
    st.session_state.input_text = ""
if "df" not in st.session_state:
    st.session_state.df = None
if "last_sel" not in st.session_state:
    st.session_state.last_sel = "직접 입력"

# ===== 헤더 =====
st.title("🧠 국어 텍스트 데이터 시각화")
st.caption("드롭다운에서 작품을 고르거나 직접 입력/업로드 → ‘워드클라우드 생성’ 후, 히스토그램/파이차트도 확인하세요. JPG 저장 지원.")

# ===== 소스 선택 =====
sel = st.selectbox("텍스트 소스 선택", options=list(FULL_TEXT_SAMPLES.keys()), index=0)
if sel != st.session_state.last_sel:
    st.session_state.input_text = FULL_TEXT_SAMPLES.get(sel, "")
    st.session_state.last_sel = sel

# ===== 입력/업로드 =====
c1, c2 = st.columns([2, 1])
with c1:
    text = st.text_area(
        "분석할 한국어 텍스트 입력/붙여넣기",
        key="input_text",
        height=240,
        placeholder="예: 시/소설 원문, 학생 글, 기사 본문 등"
    )
with c2:
    up = st.file_uploader("또는 .txt 업로드", type=["txt"])
    if up is not None:
        try:
            st.session_state.input_text = up.read().decode("utf-8", errors="ignore")
            st.success("텍스트 파일을 불러왔습니다.")
        except Exception as e:
            st.error(f"파일 읽기 오류: {e}")

# ===== 옵션 =====
default_stopwords = "이, 그, 저, 등, 데, 수, 등등, 것, 거, 있다, 없다, 하다, 되다, 아니다, 그러나, 그리고, 또한, 왜냐하면, 때문에, 은, 는, 이, 가, 을, 를, 과, 와, 도, 에, 의, 로, 으로, 에서, 보다, 까지, 만큼, 에게, 한, 듯, 싶다"
stop_input = st.text_input("불용어(쉼표로 구분)", value=default_stopwords)
stopwords = {w.strip() for w in stop_input.split(",") if w.strip()}
min_len = st.slider("포함할 최소 글자 수", 1, 4, 2, 1)
topk = st.slider("상위 몇 개 단어 시각화", 10, 200, 100, 10)

# ===== 공통 유틸 (JPG 저장) =====
def fig_to_jpg_bytes(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="jpg", bbox_inches="tight", dpi=220)
    buf.seek(0)
    return buf.read()

def pil_to_jpg_bytes(pil_image) -> bytes:
    buf = io.BytesIO()
    pil_image.save(buf, format="JPEG", quality=95)
    buf.seek(0)
    return buf.read()

# ===== 분석 함수 =====
def analyze_text(text_str: str, min_len: int, stopwords: set) -> pd.DataFrame:
    tokens = tokenize_nouns(text_str)
    freq = {}
    for w in tokens:
        if len(w) >= min_len and (w not in stopwords):
            freq[w] = freq.get(w, 0) + 1
    df = pd.DataFrame(sorted(freq.items(), key=lambda x: x[1], reverse=True), columns=["단어", "빈도"])
    return df

# ===== 1) 워드클라우드 + 빈도표 =====
make_wc = st.button("☁️ 워드클라우드 생성 (빈도분석 포함)")
if make_wc:
    if not st.session_state.input_text.strip():
        st.warning("텍스트를 입력(또는 붙여넣기/업로드)하세요.")
    else:
        df = analyze_text(st.session_state.input_text, min_len=min_len, stopwords=stopwords)
        st.session_state.df = df

        if df.empty:
            st.info("추출된 단어가 없습니다. 불용어/최소 글자 수를 조정해 보세요.")
        else:
            st.subheader("📊 단어 빈도 Top-N")
            st.dataframe(df.head(topk), use_container_width=True)

            # 워드클라우드
            wc = WordCloud(font_path=FONT_PATH, background_color="white", width=1200, height=800)\
                    .generate_from_frequencies(dict(zip(df["단어"], df["빈도"])))
            st.subheader("☁️ 워드클라우드")
            wc_img = wc.to_image()
            st.image(wc_img, use_column_width=True)
            st.download_button("워드클라우드 JPG 저장", data=pil_to_jpg_bytes(wc_img),
                               file_name="wordcloud.jpg", mime="image/jpeg")

# ===== 2) 추가 시각화 =====
st.markdown("---")
st.subheader("📈 추가 시각화")
cv1, cv2 = st.columns(2)
with cv1:
    show_hist = st.button("히스토그램(막대그래프) 보기")
with cv2:
    show_pie = st.button("파이차트 보기")

if show_hist:
    df = st.session_state.df
    if df is None or df.empty:
        st.warning("먼저 ‘워드클라우드 생성’을 눌러 빈도표를 만들어 주세요.")
    else:
        plot_df = df.head(topk)
        st.write("상위 단어 빈도 막대그래프")
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.bar(plot_df["단어"], plot_df["빈도"])
        ax.set_xticklabels(plot_df["단어"], rotation=45, ha="right")
        ax.set_ylabel("빈도")
        ax.set_xlabel("단어")
        ax.set_title("단어 빈도(Top-N) 막대그래프")
        st.pyplot(fig, use_container_width=True)
        st.download_button("히스토그램 JPG 저장", data=fig_to_jpg_bytes(fig),
                           file_name="histogram.jpg", mime="image/jpeg")

if show_pie:
    df = st.session_state.df
    if df is None or df.empty:
        st.warning("먼저 ‘워드클라우드 생성’을 눌러 빈도표를 만들어 주세요.")
    else:
        plot_df = df.head(topk)
        st.write("상위 단어 비율 파이차트")
        fig2, ax2 = plt.subplots(figsize=(6, 6))
        ax2.pie(plot_df["빈도"], labels=plot_df["단어"], autopct="%1.1f%%", startangle=90)
        ax2.axis("equal")
        ax2.set_title("단어 비율(Top-N) 파이차트")
        st.pyplot(fig2, use_container_width=True)
        st.download_button("파이차트 JPG 저장", data=fig_to_jpg_bytes(fig2),
                           file_name="piechart.jpg", mime="image/jpeg")

st.markdown("---")
st.caption("Kiwi 기반 한국어 형태소 분석(명사 추출)으로 동작합니다. 한글 폰트는 나눔고딕으로 고정했습니다.")
