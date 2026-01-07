import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, date, timedelta
import io
import calendar

# Page configuration
st.set_page_config(
    page_title="Stock Minimum Filter",
    page_icon="📈",
    layout="wide"
)

# Dynamic quarter calculation functions
def get_current_quarter():
    """Get current quarter based on today's date"""
    today = date.today()
    month = today.month
    year = today.year

    if month <= 3:
        quarter = 1
        start_month, end_month = 1, 3
    elif month <= 6:
        quarter = 2
        start_month, end_month = 4, 6
    elif month <= 9:
        quarter = 3
        start_month, end_month = 7, 9
    else:
        quarter = 4
        start_month, end_month = 10, 12

    # Get last day of the quarter
    last_day = calendar.monthrange(year, end_month)[1]

    return {
        'quarter': quarter,
        'year': year,
        'start_date': date(year, start_month, 1),
        'end_date': date(year, end_month, last_day),
        'label': f'Q{quarter} {year}'
    }

def get_previous_quarter(current_quarter_info):
    """Get previous quarter"""
    quarter = current_quarter_info['quarter']
    year = current_quarter_info['year']

    if quarter == 1:
        prev_quarter = 4
        prev_year = year - 1
    else:
        prev_quarter = quarter - 1
        prev_year = year

    if prev_quarter == 1:
        start_month, end_month = 1, 3
    elif prev_quarter == 2:
        start_month, end_month = 4, 6
    elif prev_quarter == 3:
        start_month, end_month = 7, 9
    else:
        start_month, end_month = 10, 12

    last_day = calendar.monthrange(prev_year, end_month)[1]

    return {
        'quarter': prev_quarter,
        'year': prev_year,
        'start_date': date(prev_year, start_month, 1),
        'end_date': date(prev_year, end_month, last_day),
        'label': f'Q{prev_quarter} {prev_year}'
    }

# Get dynamic quarter information
current_quarter = get_current_quarter()
previous_quarter = get_previous_quarter(current_quarter)

st.title(f"📈 Prezzo Attuale Vs. al Min/Max del Trimestre Precedente")
st.markdown(f"Trova azioni il cui **prezzo di chiusura del {current_quarter['label']}** è vicino al **minimo del {previous_quarter['label']}**")

# Default stock symbols (popular stocks)
DEFAULT_SYMBOLS = [
    # Technology
    "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA", "NFLX", 
    "AMD", "INTC", "CRM", "ORCL", "ADBE", "PYPL", "IBM", "CSCO", "NOW", "SNOW", 
    # Financial Services
    "V", "MA", "JPM", "BAC", "WFC", "GS", "MS", "C", "AXP",
    # Healthcare & Pharmaceuticals
    "JNJ", "PFE", "UNH", "ABBV", "MRK", "TMO", "ABT", "CVS", "AMGN", "GILD",
    # Consumer & Retail
    "DIS", "KO", "PEP", "WMT", "HD", "MCD", "SBUX", "NKE", "TGT", "COST",
    # Industrial & Energy
    "BA", "CAT", "GE", "MMM", "XOM", "CVX", "COP", "SLB", "EOG", "HAL",
    # Telecom & Media
    "VZ", "T", "TMUS", "CMCSA", "CHTR", "PSKY", "WBD", "FOXA",
    # Real Estate & REITs
    "SPY", "QQQ", "IWM", "VTI", "REIT", "VNQ", "AMT", "CCI", "EQIX", "PLD",
    # Emerging & Growth
    "ROKU", "SHOP", "ZM", "DOCU", "CRM", "OKTA", "TWLO", "NET", "DDOG"
]

# Sidebar configuration
st.sidebar.header("Configurazione Filtro")

# Option to upload symbols from file
st.sidebar.subheader("📁 Carica Simboli")
uploaded_file = st.sidebar.file_uploader(
    "Carica file TXT con simboli azionari:",
    type=['txt'],
    help="Carica un file .txt con simboli azionari (uno per riga o separati da virgola)"
)

# Load symbols from file or use default
if uploaded_file is not None:
    try:
        # Read the uploaded file
        file_content = uploaded_file.read().decode('utf-8')

        # Parse symbols (handle both newlines and commas)
        file_symbols = []
        for line in file_content.strip().split('\n'):
            # Split by comma and clean each symbol
            line_symbols = [s.strip().upper() for s in line.split(',') if s.strip()]
            file_symbols.extend(line_symbols)

        # Remove duplicates while preserving order
        symbols = list(dict.fromkeys(file_symbols))
        st.sidebar.success(f"✅ Caricati {len(symbols)} simboli dal file")
        st.sidebar.info(f"Simboli: {', '.join(symbols[:10])}{'...' if len(symbols) > 10 else ''}")

    except Exception as e:
        st.sidebar.error(f"❌ Errore nella lettura del file: {str(e)}")
        symbols = DEFAULT_SYMBOLS
        st.sidebar.info(f"🔍 Utilizzo dei {len(DEFAULT_SYMBOLS)} stock predefiniti")
else:
    symbols = DEFAULT_SYMBOLS
    st.sidebar.info(f"🔍 Analisi automatica di tutti i {len(DEFAULT_SYMBOLS)} stock predefiniti")
    st.sidebar.write("Oppure carica un file TXT sopra per utilizzare simboli personalizzati")

# ⚙️ Dynamic period configuration
st.sidebar.subheader("📅 Selezione Periodo")
st.sidebar.write(f"**Trimestre corrente**: {current_quarter['label']}")
st.sidebar.write(f"**Periodo**: {current_quarter['start_date'].strftime('%d/%m/%Y')} - {current_quarter['end_date'].strftime('%d/%m/%Y')}")

st.sidebar.write(f"**Trimestre precedente**: {previous_quarter['label']}")
st.sidebar.write(f"**Periodo**: {previous_quarter['start_date'].strftime('%d/%m/%Y')} - {previous_quarter['end_date'].strftime('%d/%m/%Y')}")

st.sidebar.info(f"🔄 **Confronto automatico**: Close {current_quarter['label']} vs Min {previous_quarter['label']}")
st.sidebar.caption(f"Confronta il prezzo di chiusura del trimestre corrente con il minimo del trimestre precedente")
st.sidebar.divider()

# Threshold configuration
st.sidebar.subheader("🎯 Impostazioni Filtro")
st.sidebar.write(f"**Soglia di vicinanza tra close corrente e minimo precedente:**")
threshold_percent = st.sidebar.slider(
    "Percentuale massima di differenza (%)",
    min_value=1.0,
    max_value=10.0,
    value=3.5,
    step=0.5,
    help=f"Le azioni con close del trimestre corrente entro questa percentuale dal minimo del trimestre precedente verranno mostrate"
)
st.sidebar.write(f"**Attualmente: {threshold_percent}%** differenza massima")
st.sidebar.divider()

# Set dynamic date ranges
current_start = current_quarter['start_date']
current_end = current_quarter['end_date']
previous_start = previous_quarter['start_date']
previous_end = previous_quarter['end_date']

@st.cache_data
def fetch_stock_data(symbol, start_date, end_date):
    """Fetch stock data for given symbol and date range"""
    try:
        ticker = yf.Ticker(symbol)
        data = ticker.history(start=start_date, end=end_date, auto_adjust=False) # data = ticker.history(start=start_date, end=end_date)
        if data.empty:
            return None, f"No data available for {symbol}"
        return data, None
    except Exception as e:
        return None, f"Error fetching {symbol}: {str(e)}"

def analyze_stock(symbol):
    """Analizza una singola azione confrontando il close del trimestre corrente con min/max trimestre precedente"""

    # Correggi end_date per includere l'ultimo giorno nel download yfinance
    current_end_adj = current_end + timedelta(days=1)
    previous_end_adj = previous_end + timedelta(days=1)

    # Recupera dati trimestre corrente
    current_data, curr_error = fetch_stock_data(symbol, current_start, current_end_adj)
    if curr_error or current_data.empty:
        return {"symbol": symbol, "error": f"No data available for current quarter {current_quarter['label']}: {curr_error}"}

    # Recupera dati trimestre precedente
    previous_data, prev_error = fetch_stock_data(symbol, previous_start, previous_end_adj)
    if prev_error or previous_data.empty:
        return {"symbol": symbol, "error": f"No data available for previous quarter {previous_quarter['label']}: {prev_error}"}

    # --- Trimestre corrente ---
    current_close = current_data['Close'].iloc[-1]   # ultimo close
    current_min = current_data['Low'].min()
    current_max = current_data['High'].max()
    current_avg = current_data['Close'].mean()

    # --- Trimestre precedente ---
    # Evita NaN e verifica dati
    previous_data_valid = previous_data.dropna(subset=['Low', 'High', 'Close'])
    if previous_data_valid.empty:
        return {"symbol": symbol, "error": f"No valid data for previous quarter {previous_quarter['label']}"}

    previous_min = previous_data_valid['Low'].min()
    previous_max = previous_data_valid['High'].max()
    previous_avg = previous_data_valid['Close'].mean()
    previous_min_date = previous_data_valid['Low'].idxmin()
    previous_max_date = previous_data_valid['High'].idxmax()

    # Calcolo differenza percentuale rispetto al minimo
    percent_difference = ((current_close - previous_min) / previous_min) * 100 if previous_min > 0 else 0

    # Vicinanza al minimo
    are_close = abs(percent_difference) <= threshold_percent

    # Posizione rispetto al minimo precedente
    if current_close > previous_min:
        position = "Sopra"
    elif current_close < previous_min:
        position = "Sotto"
    else:
        position = "Uguale"

    return {
        "symbol": symbol,
        "current_close": current_close,
        "current_min": current_min,
        "current_max": current_max,
        "current_avg": current_avg,
        "previous_min": previous_min,
        "previous_max": previous_max,
        "previous_avg": previous_avg,
        "previous_min_date": previous_min_date,
        "previous_max_date": previous_max_date,
        "percent_difference": percent_difference,
        "are_close": are_close,
        "position": position,
        "current_data": current_data,
        "previous_data": previous_data,
        "error": None
    }


# Main analysis - runs automatically
if symbols:
    with st.spinner(f"Analisi di {len(symbols)} azioni - confronto close {current_quarter['label']} vs minimo {previous_quarter['label']}..."):
        results = []
        progress_bar = st.progress(0)
        status_text = st.empty()

        for i, symbol in enumerate(symbols):
            status_text.text(f"Analisi {symbol}... ({i+1}/{len(symbols)})")
            result = analyze_stock(symbol)
            results.append(result)
            progress_bar.progress((i + 1) / len(symbols))

        progress_bar.empty()
        status_text.empty()

        # Filter successful results
        successful_results = [r for r in results if r.get("error") is None]
        error_results = [r for r in results if r.get("error") is not None]

        # Display errors if any
        if error_results:
            st.warning("⚠️ Errori per alcuni simboli:")
            error_data = [(r["symbol"], r["error"]) for r in error_results]
            error_df = pd.DataFrame(error_data, columns=["Simbolo", "Errore"])
            st.dataframe(error_df, use_container_width=True)
                
        if successful_results:
            # Crea DataFrame completo dei risultati
            results_data = []
            for r in successful_results:
                results_data.append({
                    "Simbolo": r["symbol"],
                    f"Close {current_quarter['label']}": f"${r['current_close']:.2f}",
                    f"Min {previous_quarter['label']}": f"${r['previous_min']:.2f}",
                    "Differenza %": f"{r['percent_difference']:.1f}%",
                    "Posizione": r["position"],
                    "Vicini": "✅" if r['are_close'] else "❌"
                })
            results_df = pd.DataFrame(results_data)

            # Filtra solo azioni vicine al minimo precedente
            close_results = [r for r in successful_results if r['are_close']]

            if close_results:
                st.subheader(f"🎯 Close {current_quarter['label']} Vicino al Minimo {previous_quarter['label']} (entro {threshold_percent}%)")

                # Metriche riassuntive
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("🎯 Close Vicini al Minimo Precedente", len(close_results))
                with col2:
                    st.metric("📊 Totali Analizzati", len(successful_results))
                with col3:
                    st.metric("📅 Confronto", f"Close {current_quarter['label']} vs Min {previous_quarter['label']}")

                # Prepara dati filtrati per tabella
                filtered_data = []
                for r in close_results:
                    row_data = {
                        "Simbolo": r["symbol"],
                        f"Min {previous_quarter['label']}": f"${r['previous_min']:.2f}",
                        f"Close {current_quarter['label']}": f"${r['current_close']:.2f}",
                        "Differenza %": f"{r['percent_difference']:.1f}%",
                        "Posizione": r["position"],
                        f"Data Min {previous_quarter['label']}": r['previous_min_date'].strftime('%d/%m/%Y')
                    }
                    filtered_data.append(row_data)

                filtered_df = pd.DataFrame(filtered_data)

                # Ordina colonne
                cols_order = [
                    "Simbolo",
                    f"Min {previous_quarter['label']}",
                    f"Close {current_quarter['label']}",
                    "Differenza %",
                    "Posizione",
                    f"Data Min {previous_quarter['label']}"
                ]
                filtered_df = filtered_df[cols_order]

                # Ordina per Differenza % crescente
                filtered_df["Differenza %"] = filtered_df["Differenza %"].str.replace("%", "").astype(float)
                filtered_df = filtered_df.sort_values(by="Differenza %", ascending=True)

                # Mostra tabella
                #st.dataframe(filtered_df, use_container_width=True)

                # --------------------------------------------------------
                # --------------------------------------------------------
                # Riduci Differenza % a 1 decimale
                filtered_df["Differenza %"] = filtered_df["Differenza %"].astype(float).round(1)

                # Funzione per evidenziare Differenza %
                def highlight_diff_text(val):
                    if val < -1:          # Differenza sotto -1%
                        return 'color: red; font-weight: bold'
                    elif val > 1:         # Differenza sopra +1%
                        return 'color: orange; font-weight: bold'
                    else:                 # Tra -1% e +1%
                        return ''

                # Applica stile
                styled_df = filtered_df.style.applymap(highlight_diff_text, subset=["Differenza %"]) \
                                             .format({"Differenza %": "{:.1f}%"})

                st.dataframe(styled_df, use_container_width=True)


                # --------------------------------------------------------
                # --------------------------------------------------------
                
                # Bottone per esportazione CSV
                csv_buffer = io.StringIO()
                filtered_df.to_csv(csv_buffer, index=False)
                csv_data = csv_buffer.getvalue()

                st.download_button(
                    label="📥 Scarica Risultati Filtrati (CSV)",
                    data=csv_data,
                    file_name=f"close_vicino_minimo_{current_quarter['label'].lower().replace(' ', '_')}_{previous_quarter['label'].lower().replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )

            # ----------------------------
            # SECONDA TABELLA (Close corrente vs Massimo trimestre precedente)
            # ----------------------------
            max_results = [r for r in successful_results if abs((r['current_close'] - r['previous_max']) / r['previous_max'] * 100) <= threshold_percent]

            if max_results:
                st.subheader(f"🎯 Close {current_quarter['label']} Vicino al Massimo {previous_quarter['label']} (entro {threshold_percent}%)")

                # Summary metrics per risultati filtrati
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("🎯 Close Vicini al Massimo Precedente", len(max_results))
                with col2:
                    st.metric("📊 Totali Analizzati", len(successful_results))
                with col3:
                    st.metric("📅 Confronto", f"Close {current_quarter['label']} vs Max {previous_quarter['label']}")

                # Prepara i dati filtrati
                filtered_max_data = []
                for r in max_results:
                    percent_diff = (r['current_close'] - r['previous_max']) / r['previous_max'] * 100
                    position = "Sopra" if r['current_close'] > r['previous_max'] else "Sotto" if r['current_close'] < r['previous_max'] else "Uguale"
                    previous_max_date = r['previous_data']['High'].idxmax()  # Data del massimo
                    row_data = {
                        "Simbolo": r["symbol"],
                        f"Max {previous_quarter['label']}": f"${r['previous_max']:.2f}",
                        f"Close {current_quarter['label']}": f"${r['current_close']:.2f}",
                        "Differenza %": f"{percent_diff:.1f}%",
                        "Posizione": position,
                        f"Data Max {previous_quarter['label']}": previous_max_date.strftime('%d/%m/%Y')
                    }
                    filtered_max_data.append(row_data)

                filtered_max_df = pd.DataFrame(filtered_max_data)

                # Ordina colonne
                cols_order_max = [
                    "Simbolo",
                    f"Max {previous_quarter['label']}",        
                    f"Close {current_quarter['label']}",       
                    "Differenza %",
                    "Posizione",
                    f"Data Max {previous_quarter['label']}"
                ]
                filtered_max_df = filtered_max_df[cols_order_max]

                # Ordina per Differenza % crescente
                filtered_max_df["Differenza %"] = filtered_max_df["Differenza %"].str.replace("%", "").astype(float)
                filtered_max_df = filtered_max_df.sort_values(by="Differenza %", ascending=True)
                

                # Mostra tabella
                #st.dataframe(filtered_max_df, use_container_width=True)

                # --------------------------------------------------------
                # --------------------------------------------------------
                # Funzione per evidenziare il testo della Differenza %
                # Riduci Differenza % a 1 decimale
                filtered_max_df["Differenza %"] = filtered_max_df["Differenza %"].astype(float).round(1)

                # Funzione per evidenziare Differenza % (solo per questa tabella)
                def highlight_diff_text(val):
                    if val < -1:          # Differenza sotto -1%
                        return 'color: red; font-weight: bold'
                    elif val > 1:         # Differenza sopra +1%
                        return 'color: orange; font-weight: bold'
                    else:                 # Tra -1% e +1%
                        return ''

                # Applica stile alla colonna Differenza %
                styled_max_df = filtered_max_df.style.applymap(highlight_diff_text, subset=["Differenza %"]) \
                                                     .format({"Differenza %": "{:.1f}%"})

                st.dataframe(styled_max_df, use_container_width=True)


                # --------------------------------------------------------
                # --------------------------------------------------------

                # Export CSV
                csv_buffer_max = io.StringIO()
                filtered_max_df.to_csv(csv_buffer_max, index=False)
                csv_data_max = csv_buffer_max.getvalue()

                st.download_button(
                    label="📥 Scarica Risultati Close vs Massimo (CSV)",
                    data=csv_data_max,
                    file_name=f"close_vicino_massimo_{current_quarter['label'].lower().replace(' ', '_')}_{previous_quarter['label'].lower().replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
            else:
                st.info(f"❌ Nessuna azione trovata con close del {current_quarter['label']} entro {threshold_percent}% dal massimo del {previous_quarter['label']}.")

            # ----------------------------    

            # Analisi grafico individuale per Close vs Minimi
            selected_stock = st.selectbox(
                "Seleziona un'azione per visualizzare il grafico dettagliato:",
                options=[r["symbol"] for r in close_results]
            )

            if selected_stock:
                selected_result = next(r for r in close_results if r["symbol"] == selected_stock)

                fig = go.Figure()
                # Candlestick trimestre precedente
                fig.add_trace(go.Candlestick(
                    x=selected_result["previous_data"].index,
                    open=selected_result["previous_data"]['Open'],
                    high=selected_result["previous_data"]['High'],
                    low=selected_result["previous_data"]['Low'],
                    close=selected_result["previous_data"]['Close'],
                    name=f"{previous_quarter['label']}",
                    increasing_line_color='green',
                    decreasing_line_color='red'
                ))
                # Candlestick trimestre corrente
                fig.add_trace(go.Candlestick(
                    x=selected_result["current_data"].index,
                    open=selected_result["current_data"]['Open'],
                    high=selected_result["current_data"]['High'],
                    low=selected_result["current_data"]['Low'],
                    close=selected_result["current_data"]['Close'],
                    name=f"{current_quarter['label']}",
                    increasing_line_color='blue',
                    decreasing_line_color='orange'
                ))

                # Linee orizzontali
                fig.add_hline(y=selected_result["previous_min"], line_dash="dash", line_color="red",
                              annotation_text=f"Min {previous_quarter['label']}")
                fig.add_hline(y=selected_result["current_close"], line_dash="solid", line_color="green",
                              annotation_text=f"Close {current_quarter['label']}")

                fig.update_layout(
                    title=f'{selected_stock} - Confronto Trimestrale',
                    xaxis_title='Data',
                    yaxis_title='Prezzo ($)',
                    showlegend=True
                )
                st.plotly_chart(fig, use_container_width=True)

                # Metriche individuali
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric(f"Close {current_quarter['label']}", f"${selected_result['current_close']:.2f}")
                with col2:
                    st.metric(f"Min {previous_quarter['label']}", f"${selected_result['previous_min']:.2f}")
                with col3:
                    st.metric("Differenza %", f"{selected_result['percent_difference']:.1f}%")
                with col4:
                    st.metric("Posizione", selected_result['position'])
            else:
                st.info(f"❌ Nessuna azione trovata con close del {current_quarter['label']} entro {threshold_percent}% dal minimo del {previous_quarter['label']}.")
                st.markdown(f"**Analizzate {len(successful_results)} azioni** - Prova ad aumentare la percentuale di tolleranza o selezionare azioni diverse.")

        else:
            st.error("Impossibile recuperare dati per nessuna delle azioni specificate. Controlla i simboli e riprova.")

        # ----------------------------
        
        # Analisi grafico individuale per Close vs Massimo
        selected_stock_max = st.selectbox(
            "Seleziona un'azione per visualizzare il grafico Close vs Massimo:",
            options=[r["symbol"] for r in max_results]
        )

        if selected_stock_max:
            selected_result_max = next(r for r in max_results if r["symbol"] == selected_stock_max)

            fig_max = go.Figure()
            # Candlestick trimestre precedente
            fig_max.add_trace(go.Candlestick(
                x=selected_result_max["previous_data"].index,
                open=selected_result_max["previous_data"]['Open'],
                high=selected_result_max["previous_data"]['High'],
                low=selected_result_max["previous_data"]['Low'],
                close=selected_result_max["previous_data"]['Close'],
                name=f"{previous_quarter['label']}",
                increasing_line_color='green',
                decreasing_line_color='red'
            ))
            # Candlestick trimestre corrente
            fig_max.add_trace(go.Candlestick(
                x=selected_result_max["current_data"].index,
                open=selected_result_max["current_data"]['Open'],
                high=selected_result_max["current_data"]['High'],
                low=selected_result_max["current_data"]['Low'],
                close=selected_result_max["current_data"]['Close'],
                name=f"{current_quarter['label']}",
                increasing_line_color='blue',
                decreasing_line_color='orange'
            ))

            # Linee orizzontali
            fig_max.add_hline(y=selected_result_max["previous_max"], line_dash="dash", line_color="red",
                              annotation_text=f"Max {previous_quarter['label']}")
            fig_max.add_hline(y=selected_result_max["current_close"], line_dash="solid", line_color="green",
                              annotation_text=f"Close {current_quarter['label']}")

            fig_max.update_layout(
                title=f'{selected_stock_max} - Confronto Close Corrente vs Massimo Precedente',
                xaxis_title='Data',
                yaxis_title='Prezzo ($)',
                showlegend=True
            )
            st.plotly_chart(fig_max, use_container_width=True)

            # Metriche individuali
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric(f"Close {current_quarter['label']}", f"${selected_result_max['current_close']:.2f}")
            with col2:
                st.metric(f"Max {previous_quarter['label']}", f"${selected_result_max['previous_max']:.2f}")
            with col3:
                st.metric("Differenza %", f"{selected_result_max['percent_difference']:.1f}%")
            with col4:
                st.metric("Posizione", selected_result_max['position'])


# Information section
with st.expander("ℹ️ Come funziona"):
    st.markdown(f"""
    ### Processo di Analisi:
    1. **Raccolta Dati**: Recupera i dati storici per **{current_quarter['label']}** e **{previous_quarter['label']}**
    2. **Calcolo Minimo Precedente**: Identifica il prezzo minimo del trimestre precedente
    3. **Calcolo Close Corrente**: Prende l'ultimo prezzo di chiusura del trimestre corrente
    4. **Confronto**: Calcola la differenza percentuale tra il close corrente e il minimo precedente
    5. **Filtraggio**: Mostra le azioni i cui valori sono vicini secondo la soglia impostata

    ### Metriche Chiave:
    - **Close {current_quarter['label']}**: Ultimo prezzo di chiusura del trimestre corrente
    - **Min {previous_quarter['label']}**: Prezzo minimo del trimestre precedente
    - **Differenza %**: Variazione percentuale tra i due valori
    - **Posizione**: Se il close corrente è sopra o sotto il minimo precedente

    ### Caratteristiche Dinamiche:
    - **Rilevamento Automatico Trimestri**: Analizza sempre i trimestri correnti
    - **Confronto Sequenziale**: Confronta sempre con il trimestre immediatamente precedente
    - **Logica Trimestrale Intelligente**: Q3→Q2, Q4→Q3, Q2→Q1, Q1→Q4 (anno precedente)
    - **Aggiornamenti in Tempo Reale**: Nessun aggiornamento manuale richiesto

    ### Casi d'Uso:
    - **Analisi Tecnica**: Identifica azioni che potrebbero aver trovato supporto ai minimi precedenti
    - **Trading Range**: Trova azioni che stanno testando livelli di supporto precedenti
    - **Analisi di Tendenza**: Valuta se i supporti storici stanno reggendo
    """)

st.markdown("---")
st.markdown("📊 Dati forniti da Yahoo Finance tramite libreria yfinance")