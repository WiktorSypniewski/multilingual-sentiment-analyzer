import streamlit as st
import pandas as pd
import torch
import io
import matplotlib.pyplot as plt
from deep_translator import GoogleTranslator
from model import BGERidgePredictor

try:
    torch.classes.__path__ = []
except Exception:
    pass

st.set_page_config(page_title="Analizator Sentymentu", layout="wide")

@st.cache_resource
def load_model():
    predictor = BGERidgePredictor(model_path="regresor_ridge_bge.joblib")
    predictor.load_resources()
    return predictor

def translate_to_english(text):
    """Funkcja tłumacząca tekst na angielski przy użyciu darmowego API (Google Translate)."""
    try:
        return GoogleTranslator(source='pl', target='en').translate(text)
    except Exception as e:
        return f"Translation error: {e}"

def generate_html_report(df_display):
    df_clean = df_display.copy()
    df_clean['SENTYMENTY'] = df_clean['SENTYMENTY'].str.replace('✅ ', '').str.replace('⚠️ ', '').str.replace('❌ ', '')
    
    sentiment_counts = df_clean['SENTYMENTY'].value_counts()
    avg_score = df_display['Wynik'].mean()
    min_score = df_display['Wynik'].min()
    max_score = df_display['Wynik'].max()
    
    total = len(df_display)
    pozytywne_pct = (sentiment_counts.get('POZYTYWNA', 0) / total) * 100
    mieszane_pct = (sentiment_counts.get('MIESZANA', 0) / total) * 100
    negatywne_pct = (sentiment_counts.get('NEGATYWNA', 0) / total) * 100
    
    # Sprawdzenie czy mamy kolumnę z tłumaczeniem
    has_translation = 'Opinia (EN)' in df_display.columns
    
    opinions_html = ""
    
    # Słownik do tłumaczenia etykiet w tabeli HTML
    sentiment_en_map = {'POZYTYWNA': 'POSITIVE', 'MIESZANA': 'MIXED', 'NEGATYWNA': 'NEGATIVE'}
    
    for _, row in df_display.iterrows():
        sentiment = row['SENTYMENTY']
        color = '#4CAF50' if '✅' in sentiment else ('#FFC107' if '⚠️' in sentiment else '#F44336')
        
        raw_sent = sentiment.replace('✅ ', '').replace('⚠️ ', '').replace('❌ ', '')
        emoji = sentiment.split()[0] if len(sentiment.split()) > 0 else ""
        en_sent = f"{emoji} {sentiment_en_map.get(raw_sent, raw_sent)}"
        
        translation_cell = f"<td>{row['Opinia (EN)']}</td>" if has_translation else ""
        
        opinions_html += f"""
        <tr>
            <td>{row['Wynik']:.2f}</td>
            <td><span style="background-color: {color}; color: white; padding: 5px 10px; border-radius: 5px; font-weight: bold;">{en_sent}</span></td>
            <td>{row.get('Opinia (PL)', row.get('Opinia', ''))}</td>
            {translation_cell}
        </tr>
        """
        
    translation_header = '<th style="width: 35%;">English Translation</th>' if has_translation else ""
    opinia_width = "40%" if has_translation else "77%"
    
    html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Sentiment Analysis Report</title>
        <style>
            * {{ margin: 0; padding: 0; box-sizing: border-box; }}
            @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700;800&family=Inter:wght@300;400;500;600&display=swap');
            body {{ font-family: 'Inter', sans-serif; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 40px 20px; min-height: 100vh; }}
            .container {{ max-width: 1400px; margin: 0 auto; background: white; border-radius: 20px; box-shadow: 0 30px 80px rgba(0,0,0,0.25); overflow: hidden; }}
            .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 60px 40px; text-align: center; position: relative; overflow: hidden; }}
            .header h1 {{ font-family: 'Poppins', sans-serif; font-size: 48px; margin-bottom: 10px; font-weight: 800; }}
            .header p {{ font-size: 18px; opacity: 0.95; font-weight: 300; }}
            .content {{ padding: 50px 40px; }}
            .stats {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 25px; margin-bottom: 50px; }}
            .stat-box {{ background: linear-gradient(135deg, #f5f7fa 0%, #e9ecef 100%); padding: 30px; border-radius: 15px; border-left: 5px solid #667eea; text-align: center; }}
            .stat-value {{ font-size: 40px; font-weight: 800; color: #667eea; margin-bottom: 10px; font-family: 'Poppins', sans-serif; }}
            .stat-label {{ color: #666; font-size: 13px; text-transform: uppercase; font-weight: 600; }}
            .sentiment-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 25px; margin-bottom: 50px; }}
            .sentiment-card {{ padding: 30px; border-radius: 15px; color: white; text-align: center; font-weight: 600; box-shadow: 0 10px 30px rgba(0,0,0,0.15); }}
            .sentiment-card.positive {{ background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%); }}
            .sentiment-card.neutral {{ background: linear-gradient(135deg, #FFC107 0%, #e0a800 100%); color: #333; }}
            .sentiment-card.negative {{ background: linear-gradient(135deg, #F44336 0%, #da190b 100%); }}
            .sentiment-card-number {{ font-size: 36px; margin-bottom: 12px; font-family: 'Poppins', sans-serif; }}
            .table-section h2 {{ color: #333; margin-bottom: 25px; padding-bottom: 15px; border-bottom: 3px solid #667eea; font-family: 'Poppins', sans-serif; }}
            table {{ width: 100%; border-collapse: collapse; background: white; border-radius: 10px; overflow: hidden; }}
            th {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 18px 15px; text-align: left; font-weight: 700; font-size: 14px; text-transform: uppercase; }}
            td {{ padding: 16px 15px; border-bottom: 1px solid #e9ecef; font-size: 14px; color: #555; }}
            tbody tr:hover {{ background: #f8f9ff; }}
            .footer {{ text-align: center; padding: 30px 40px; background: #f5f7fa; color: #666; font-size: 13px; border-top: 2px solid #e9ecef; font-weight: 500; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>📊 Sentiment Analysis Report</h1>
                <p>Detailed analysis of opinions and reviews</p>
            </div>
            
            <div class="content">
                <div class="stats">
                    <div class="stat-box"><div class="stat-value">{avg_score:.2f}</div><div class="stat-label">Average Rating</div></div>
                    <div class="stat-box"><div class="stat-value">{max_score:.2f}</div><div class="stat-label">Highest Rating</div></div>
                    <div class="stat-box"><div class="stat-value">{min_score:.2f}</div><div class="stat-label">Lowest Rating</div></div>
                    <div class="stat-box"><div class="stat-value">{total}</div><div class="stat-label">Total Reviews</div></div>
                </div>
                
                <div class="sentiment-grid">
                    <div class="sentiment-card positive">
                        <div class="sentiment-card-number">✅ {sentiment_counts.get('POZYTYWNA', 0)}</div>
                        <div>{pozytywne_pct:.1f}% Positive</div>
                    </div>
                    <div class="sentiment-card neutral">
                        <div class="sentiment-card-number">⚠️ {sentiment_counts.get('MIESZANA', 0)}</div>
                        <div>{mieszane_pct:.1f}% Mixed</div>
                    </div>
                    <div class="sentiment-card negative">
                        <div class="sentiment-card-number">❌ {sentiment_counts.get('NEGATYWNA', 0)}</div>
                        <div>{negatywne_pct:.1f}% Negative</div>
                    </div>
                </div>
                
                <div class="table-section">
                    <h2>📋 Detailed Review List</h2>
                    <table>
                        <thead>
                            <tr>
                                <th style="width: 8%;">Score</th>
                                <th style="width: 17%;">Sentiment</th>
                                <th style="width: {opinia_width};">Review (Original)</th>
                                {translation_header}
                            </tr>
                        </thead>
                        <tbody>
                            {opinions_html}
                        </tbody>
                    </table>
                </div>
            </div>
            
            <div class="footer">
                <p>✨ Generated by Sentiment Analyzer | {pd.Timestamp.now().strftime('%Y-%m-%d at %H:%M:%S')} ✨</p>
            </div>
        </div>
    </body>
    </html>
    """
    return html

def main():
    st.title(" Analizator Sentymentu z Tłumaczem AI")
    engine = load_model()

    tab1, tab2 = st.tabs(["Pojedyncza opinia", "Przetwarzanie z pliku (.txt)"])

    with tab1:
        txt_input = st.text_area("Wpisz tekst do oceny:")
        if st.button("Sprawdź", key="btn_single", type="primary"):
            if txt_input:
                score = engine.process_and_predict(txt_input)
                
                if score is not None:
                    st.write("Wynik (0-10)")
                    if score < 4.0:
                        st.markdown(f"# {score:.2f} - ❌ Negatywna recenzja")
                    elif score <= 6.0:
                        st.markdown(f"# {score:.2f} - ⚠️ Mieszana recenzja")
                    else:
                        st.markdown(f"# {score:.2f} - ✅ Pozytywna recenzja")

    with tab2:
        st.info("Wgraj plik tekstowy. Model oceni sentyment, a translator (opcjonalnie) przetłumaczy tekst na angielski.")
        
        # Opcja włączenia tłumaczenia
        use_translation = st.checkbox("🌐 Przetłumacz opinie na angielski")
        
        uploaded_file = st.file_uploader("Wybierz plik .txt", type=['txt'], key="uploader")

        if uploaded_file is not None:
            content = uploaded_file.read().decode("utf-8")
            lines = [line.strip() for line in content.splitlines() if line.strip()]
            st.write(f"Do przeanalizowania: **{len(lines)}** opinii.")

            if st.button("Uruchom analizę", key="btn_bulk", type="primary"):
                results = []
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                batch_size = 64
                total_batches = (len(lines) + batch_size - 1) // batch_size
                
                for i in range(total_batches):
                    batch = lines[i*batch_size : (i+1)*batch_size]
                    
                    status_text.text(f"Ocenianie sentymentu dla partii {i+1}/{total_batches}...")
                    scores = engine.process_and_predict_batch(batch)
                    
                    for idx, (text, score) in enumerate(zip(batch, scores)):
                        if score < 4.0:
                            status = "NEGATYWNA"
                        elif score <= 6.0:
                            status = "MIESZANA"
                        else:
                            status = "POZYTYWNA"
                        
                        row_data = {
                            "Opinia (PL)": text, 
                            "Wynik": round(score, 2),
                            "SENTYMENTY": status  
                        }
                        
                        # Wywołanie tłumacza jeśli checkbox jest zaznaczony
                        if use_translation:
                            status_text.text(f"Tłumaczenie opinii {idx + 1 + (i*batch_size)}/{len(lines)}...")
                            row_data["Opinia (EN)"] = translate_to_english(text)
                            
                        results.append(row_data)
                        
                    progress_bar.progress((i + 1) / total_batches)

                status_text.text("✅ Analiza zakończona!")
                st.session_state['wyniki_df'] = pd.DataFrame(results)

        if 'wyniki_df' in st.session_state:
            df = st.session_state['wyniki_df']
            
            df_display = df.copy()
            df_display['SENTYMENTY'] = df_display['SENTYMENTY'].apply(
                lambda x: f"✅ {x}" if x == "POZYTYWNA" else (f"⚠️ {x}" if x == "MIESZANA" else f"❌ {x}")
            )
            
            st.subheader("Wyniki")
            st.dataframe(df_display, width='stretch')
            st.divider()
            
            st.subheader("📊 Raport Analityczny")
            
            col_chart1, col_chart2 = st.columns(2)
            
            with col_chart1:
                sentiment_counts = df['SENTYMENTY'].value_counts()
                
                # Zastosowanie angielskich etykiet na wykresie
                eng_labels_map = {'POZYTYWNA': 'POSITIVE', 'MIESZANA': 'MIXED', 'NEGATYWNA': 'NEGATIVE'}
                eng_labels = [eng_labels_map.get(s, s) for s in sentiment_counts.index]
                
                colors = {'POZYTYWNA': '#4CAF50', 'MIESZANA': '#FFC107', 'NEGATYWNA': '#F44336'}
                color_list = [colors.get(sentiment, '#999') for sentiment in sentiment_counts.index]
                
                fig, ax = plt.subplots(figsize=(8, 6))
                wedges, texts, autotexts = ax.pie(
                    sentiment_counts.values, 
                    labels=eng_labels,
                    autopct='%1.1f%%',
                    colors=color_list,
                    startangle=90,
                    textprops={'fontsize': 12, 'weight': 'bold'}
                )
                
                for autotext in autotexts:
                    autotext.set_color('white')
                    autotext.set_fontsize(14)
                
                ax.set_title('Sentiment Distribution', fontsize=16, weight='bold', pad=20)
                st.pyplot(fig)
            
            with col_chart2:
                avg_score = df['Wynik'].mean()
                min_score = df['Wynik'].min()
                max_score = df['Wynik'].max()
                
                st.metric("Average rating", f"{avg_score:.2f}/10")
                st.metric("Lowest rating", f"{min_score:.2f}/10")
                st.metric("Highest rating", f"{max_score:.2f}/10")
                st.write(f"**Total reviews:** {len(df)}")
                
                st.write("**Detailed breakdown:**")
                eng_names = {'POZYTYWNA': 'POSITIVE', 'MIESZANA': 'MIXED', 'NEGATYWNA': 'NEGATIVE'}
                
                for sentiment in ['POZYTYWNA', 'MIESZANA', 'NEGATYWNA']:
                    count = (df['SENTYMENTY'] == sentiment).sum()
                    percentage = (count / len(df)) * 100
                    emoji = '✅' if sentiment == 'POZYTYWNA' else ('⚠️' if sentiment == 'MIESZANA' else '❌')
                    st.write(f"{emoji} **{eng_names[sentiment]}:** {count} ({percentage:.1f}%)")
            
            st.divider()
            st.write("**Pobierz raporty:**")
            
            col1, col2 = st.columns(2)
            col3, col4 = st.columns(2)
            
            with col1:
                buffer = io.BytesIO()
                with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                    df_display.to_excel(writer, index=False)
                st.download_button(
                    label="📥 EXCEL .xlsx", 
                    data=buffer.getvalue(), 
                    file_name="analiza_sentymentu.xlsx", 
                    mime="application/vnd.ms-excel"
                )
            
            with col2:
                if 'Opinia (EN)' in df_display.columns:
                    txt_out = "\n".join([f"{r['Wynik']} | {r['SENTYMENTY']} | {r['Opinia (PL)']} | {r['Opinia (EN)']}" for _, r in df_display.iterrows()])
                else:
                    txt_out = "\n".join([f"{r['Wynik']} | {r['SENTYMENTY']} | {r.get('Opinia (PL)', r.get('Opinia', ''))}" for _, r in df_display.iterrows()])
                
                st.download_button(
                    label="📥 TXT .txt", 
                    data=txt_out.encode('utf-8'), 
                    file_name="analiza_sentymentu.txt", 
                    mime="text/plain"
                )
            
            with col3:
                html_report = generate_html_report(df_display)
                st.download_button(
                    label="📥 HTML .html", 
                    data=html_report.encode('utf-8'), 
                    file_name="analiza_sentymentu.html", 
                    mime="text/html"
                )
            
            with col4:
                csv_out = df_display.to_csv(index=False)
                st.download_button(
                    label="📥 CSV", 
                    data=csv_out.encode('utf-8'), 
                    file_name="analiza_sentymentu.csv", 
                    mime="text/csv"
                )

if __name__ == "__main__":
    main()