"""
╔════════════════════════════════════════════════════════════════════════════════════════════════════╗
║                      PORTFOLIO OPTIMIZER - STREAMLIT WEB APP v1.0                                  ║
║                                                                                                    ║
║  Sistema professionale di ottimizzazione del portafoglio con 4 strategie quantitative,            ║
║  metriche di performance avanzate e visualizzazioni interattive.                                   ║
║                                                                                                    ║
║  Autori: Ciullo, Elisabetta, Campeggio, De Pascalis                                               ║
║  Convertito in Web App da: Claude (Anthropic)                                                      ║
╚════════════════════════════════════════════════════════════════════════════════════════════════════╝
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from pypfopt import expected_returns, risk_models, EfficientFrontier, EfficientSemivariance, HRPOpt
from scipy.optimize import minimize, Bounds
import warnings

warnings.filterwarnings('ignore')

# ════════════════════════════════════════════════════════════════════════════════
# CONFIGURAZIONE PAGINA STREAMLIT
# ════════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Portfolio Optimizer Pro",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ════════════════════════════════════════════════════════════════════════════════
# CSS PERSONALIZZATO - STILE BLOOMBERG-INSPIRED
# ════════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
    /* Font e colori base */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* Header principale */
    .main-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    
    .main-header h1 {
        color: #00d4ff;
        margin: 0;
        font-weight: 700;
        font-size: 2rem;
    }
    
    .main-header p {
        color: #94a3b8;
        margin: 0.5rem 0 0 0;
        font-size: 0.95rem;
    }
    
    /* Card metriche */
    .metric-card {
        background: linear-gradient(145deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
        transition: transform 0.2s, box-shadow 0.2s;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0,212,255,0.15);
    }
    
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #00d4ff;
    }
    
    .metric-label {
        font-size: 0.85rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* Benchmark badge */
    .benchmark-badge {
        background: linear-gradient(135deg, #059669 0%, #047857 100%);
        color: white;
        padding: 0.8rem 1.5rem;
        border-radius: 8px;
        display: inline-block;
        font-weight: 600;
        margin: 0.5rem 0;
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
    }
    
    section[data-testid="stSidebar"] .stMarkdown {
        color: #e2e8f0;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #1e293b;
        border-radius: 10px;
        padding: 5px;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        color: #94a3b8;
        font-weight: 500;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #0f3460 !important;
        color: #00d4ff !important;
    }
    
    /* DataFrame styling */
    .dataframe {
        font-size: 0.85rem;
    }
    
    /* Success/Warning/Error boxes */
    .success-box {
        background: linear-gradient(135deg, #064e3b 0%, #065f46 100%);
        border-left: 4px solid #10b981;
        padding: 1rem;
        border-radius: 0 8px 8px 0;
        margin: 1rem 0;
    }
    
    .warning-box {
        background: linear-gradient(135deg, #78350f 0%, #92400e 100%);
        border-left: 4px solid #f59e0b;
        padding: 1rem;
        border-radius: 0 8px 8px 0;
        margin: 1rem 0;
    }
    
    /* Glossario styling */
    .glossary-section {
        background: #1e293b;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
        border: 1px solid #334155;
    }
    
    .glossary-title {
        color: #00d4ff;
        font-size: 1.2rem;
        font-weight: 600;
        border-bottom: 2px solid #334155;
        padding-bottom: 0.5rem;
        margin-bottom: 1rem;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #0ea5e9 0%, #0284c7 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        transition: all 0.3s;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #38bdf8 0%, #0ea5e9 100%);
        box-shadow: 0 4px 15px rgba(14,165,233,0.4);
    }
</style>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════════
# CONFIGURAZIONI DEFAULT
# ════════════════════════════════════════════════════════════════════════════════

DEFAULT_TICKERS = """SXLK
XDWT
CSNDX
WTEC
NQSE
AIQ
WTAI
XNGI
SIXG
SEME
SMH
QTUM
FTXL
CTEK
HNSC"""

BENCHMARK_CANDIDATES = [
    {'ticker': 'SPY', 'name': 'S&P 500'},
    {'ticker': 'QQQ', 'name': 'NASDAQ 100'},
    {'ticker': 'VTI', 'name': 'Total US'},
    {'ticker': 'XLK', 'name': 'Tech SPDR'},
    {'ticker': 'SMH', 'name': 'Semiconductor'},
    {'ticker': 'VUG', 'name': 'Growth'},
]

DEFAULT_SECTOR_MAP = {
    'SXLK': 'Technology', 'XDWT': 'Technology', 'XLK': 'Technology', 'VGT': 'Technology', 'IYW': 'Technology',
    'CSNDX': 'NASDAQ', 'NQSE': 'NASDAQ', 'QQQ': 'NASDAQ', 'TQQQ': 'NASDAQ',
    'AIQ': 'AI', 'WTAI': 'AI', 'BOTZ': 'AI', 'ROBO': 'AI', 'IRBO': 'AI',
    'SMH': 'Semiconductor', 'SOXX': 'Semiconductor', 'FTXL': 'Semiconductor', 'HNSC': 'Semiconductor', 'PSI': 'Semiconductor',
    'WTEC': 'Cloud', 'SKYY': 'Cloud', 'CLOU': 'Cloud', 'XNGI': 'Cloud', 'WCLD': 'Cloud',
    'SIXG': 'EmergingTech', 'QTUM': 'EmergingTech', 'ARKQ': 'EmergingTech', 'ARKG': 'EmergingTech',
    'CTEK': 'CleanEnergy', 'ICLN': 'CleanEnergy', 'TAN': 'CleanEnergy', 'QCLN': 'CleanEnergy',
    'SEME': 'EmergingMarkets', 'EEM': 'EmergingMarkets', 'VWO': 'EmergingMarkets', 'IEMG': 'EmergingMarkets',
    'CIBR': 'Cybersecurity', 'HACK': 'Cybersecurity', 'BUG': 'Cybersecurity',
    'FINX': 'Fintech', 'ARKF': 'Fintech',
    'IBB': 'HealthTech', 'XBI': 'HealthTech',
}

DEFAULT_SECTOR_LIMITS = {
    'Technology': 0.35,
    'NASDAQ': 0.30,
    'AI': 0.25,
    'Semiconductor': 0.30,
    'Cloud': 0.25,
    'EmergingTech': 0.20,
    'CleanEnergy': 0.20,
    'EmergingMarkets': 0.15,
    'Cybersecurity': 0.20,
    'Fintech': 0.15,
    'HealthTech': 0.20,
    'Other': 0.25,
}

# ════════════════════════════════════════════════════════════════════════════════
# GLOSSARIO FINANZIARIO
# ════════════════════════════════════════════════════════════════════════════════

GLOSSARIO = """
## 📚 Glossario Completo delle Metriche Finanziarie

---

### 📈 SEZIONE 1: Metriche di Rendimento

#### 1. Rendimento Annualizzato (CAGR)
**Formula:** `CAGR = (Valore_finale / Valore_iniziale)^(1/anni) - 1`

Il tasso di crescita medio annuo composto del tuo investimento. A differenza della media semplice, tiene conto dell'effetto "interesse composto" (i guadagni che generano altri guadagni).

**Esempio:** Investi 10.000€, dopo 3 anni hai 13.310€ → CAGR = 10% annuo

**Valori di riferimento per ETF azionari:**
- **> 15%** → Eccellente (difficile da mantenere nel lungo periodo)
- **10-15%** → Ottimo
- **7-10%** → Buono (in linea con la media storica del mercato)
- **< 7%** → Sotto la media storica

---

#### 2. Rendimento in Eccesso (Excess Return)
**Formula:** `Excess Return = Rendimento_portafoglio - Tasso_risk_free`

Quanto hai guadagnato IN PIÙ rispetto a un investimento "sicuro" come i titoli di stato. È il "premio" che ricevi per aver accettato il rischio di investire in asset più volatili.

---

### ⚠️ SEZIONE 2: Metriche di Rischio

#### 3. Volatilità (Deviazione Standard Annualizzata)
**Formula:** `σ = σ_giornaliera × √252`

Misura quanto i rendimenti "oscillano" attorno alla media. Una volatilità alta significa che il valore del portafoglio può variare molto da un giorno all'altro.

**Come si legge:** Una volatilità del 20% significa che, statisticamente:
- Nel 68% dei casi, il rendimento annuo sarà tra (media-20%) e (media+20%)
- Nel 95% dei casi, sarà tra (media-40%) e (media+40%)

**Valori di riferimento:**
- **< 10%** → Bassa volatilità (obbligazioni, portafogli conservativi)
- **10-20%** → Moderata (portafogli bilanciati)
- **20-30%** → Alta (portafogli azionari aggressivi)
- **> 30%** → Molto alta (singole azioni, settori speculativi)

---

#### 4. Downside Deviation (Semi-Deviazione)
**Formula:** `DD = √[Σ(min(rᵢ - target, 0))² / n]`

Come la volatilità, ma considera SOLO le oscillazioni negative. Si basa sull'idea che agli investitori non importa della "volatilità positiva" (guadagni inaspettati), ma solo di quella negativa (perdite).

---

#### 5. Maximum Drawdown (MDD)
**Formula:** `MDD = (Picco - Minimo_successivo) / Picco × 100`

La peggior perdita percentuale che avresti subito se avessi comprato al momento peggiore (il picco) e venduto al momento peggiore (il minimo successivo).

**Valori di riferimento:**
- **< 10%** → Molto conservativo
- **10-20%** → Conservativo/Moderato
- **20-30%** → Moderato/Aggressivo
- **30-50%** → Aggressivo (tipico di portafogli 100% azionari)
- **> 50%** → Molto aggressivo

---

### 📊 SEZIONE 3: Metriche Risk-Adjusted

#### 6. Sharpe Ratio
**Formula:** `SR = (Rendimento_portafoglio - Risk_free) / Volatilità`

Misura quanti "punti percentuali" di rendimento extra ottieni per ogni "punto percentuale" di rischio (volatilità) che ti assumi.

**Valori di riferimento:**
- **< 0** → Rendimento inferiore al risk-free (pessimo)
- **0 - 0.5** → Scarso
- **0.5 - 1.0** → Accettabile
- **1.0 - 2.0** → Buono
- **> 2.0** → Eccellente (difficile da mantenere)

---

#### 7. Sortino Ratio
**Formula:** `SoR = (Rendimento_portafoglio - Risk_free) / Downside_Deviation`

Come lo Sharpe, ma usa la Downside Deviation invece della volatilità totale. Penalizza solo il rischio di perdita, non la volatilità "buona".

**Valori di riferimento:**
- **< 1.0** → Scarso
- **1.0 - 2.0** → Buono
- **> 2.0** → Ottimo
- **> 3.0** → Eccellente

---

#### 8. Calmar Ratio
**Formula:** `CR = Rendimento_annualizzato / |Maximum_Drawdown|`

Rapporto tra rendimento e peggior perdita storica.

**Valori di riferimento:**
- **< 0.5** → Il drawdown è troppo alto rispetto al rendimento
- **0.5 - 1.0** → Accettabile
- **1.0 - 2.0** → Buono
- **> 2.0** → Eccellente

---

#### 9. Treynor Ratio
**Formula:** `TR = (Rendimento_portafoglio - Risk_free) / Beta`

Simile allo Sharpe, ma usa il Beta invece della volatilità. Misura il rendimento extra per unità di RISCHIO SISTEMATICO.

---

### 🎯 SEZIONE 4: Metriche Relative al Benchmark

#### 10. Beta (β)
**Formula:** `β = Covarianza(Rp, Rb) / Varianza(Rb)`

Misura la sensibilità del portafoglio ai movimenti del mercato.

**Come si legge:**
- **β = 1.0** → Il portafoglio si muove esattamente come il mercato
- **β = 1.5** → Movimenti amplificati del 50%
- **β = 0.5** → Movimenti dimezzati rispetto al mercato
- **β = 0** → Nessuna correlazione con il mercato

---

#### 11. Alpha (α)
**Formula:** `α = Rendimento_portafoglio - Rendimento_benchmark`

L'extra-rendimento generato rispetto al benchmark. È il "valore aggiunto" della gestione.

---

#### 12. Tracking Error (TE)
**Formula:** `TE = Dev_Standard(Rp - Rb) × √52`

Misura quanto il portafoglio "devia" dal benchmark nel tempo.

**Valori di riferimento:**
- **< 2%** → Gestione passiva (replica l'indice)
- **2-5%** → Gestione attiva moderata
- **5-10%** → Gestione attiva significativa
- **> 10%** → Portafoglio molto diverso dal benchmark

---

#### 13. Information Ratio (IR)
**Formula:** `IR = Alpha / Tracking_Error`

Misura la QUALITÀ dell'alpha generato.

**Valori di riferimento:**
- **< 0** → Alpha negativo (sottoperformance)
- **0 - 0.5** → Scarso
- **0.5 - 1.0** → Buono
- **> 1.0** → Eccellente

---

### 🔧 SEZIONE 5: Strategie di Ottimizzazione

#### 14. Max Sharpe (Mean-Variance di Markowitz)
Massimizza lo Sharpe Ratio trovando la combinazione di pesi che offre il MIGLIOR rapporto tra rendimento atteso e rischio.

**Pro:** Approccio teoricamente ottimale (Premio Nobel 1990)
**Contro:** Molto sensibile agli errori di stima, tende a concentrarsi su pochi asset

---

#### 15. Max Sortino
Ottimizza rispetto al rischio di RIBASSO invece che al rischio totale.

**Pro:** Più adatto per investitori avversi alle perdite
**Contro:** Richiede più dati storici per stime accurate

---

#### 16. Risk Parity
Equalizza i contributi al rischio di ogni asset.

**Pro:** Portafoglio più bilanciato, meno sensibile agli errori di stima
**Contro:** Non massimizza il rendimento, può sottoperformare in bull market

---

#### 17. HRP (Hierarchical Risk Parity)
Allocazione robusta basata su clustering gerarchico.

**Pro:** NON richiede inversione della matrice di covarianza, robusto a errori di stima
**Contro:** Non ottimizza esplicitamente rendimento o Sharpe

---

### 🔒 SEZIONE 6: Vincoli di Ottimizzazione

#### 18. Vincoli sui Pesi
`min_weight ≤ wᵢ ≤ max_weight per ogni asset`

Evitano allocazioni estreme e assicurano diversificazione minima.

---

#### 19. Vincoli Settoriali
`Σ wᵢ ≤ max_sector per ogni settore`

Limitano l'esposizione a singoli settori per diversificare il rischio settoriale.

---

#### 20. Volatility Cap
`σ_portfolio ≤ target_volatility`

Mantiene il rischio complessivo sotto una soglia predefinita.
"""


# ════════════════════════════════════════════════════════════════════════════════
# FUNZIONI DI DOWNLOAD CON CACHE
# ════════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=3600, show_spinner=False)
def download_data(tickers, start_date, end_date):
    """
    Scarica i dati storici degli ETF da Yahoo Finance.
    Utilizza cache per evitare download ripetuti.
    """
    all_data = {}
    errors = []
    
    for t in tickers:
        try:
            data = yf.Ticker(t).history(start=start_date, end=end_date, auto_adjust=True)
            if data.empty or len(data) < 50:
                # Prova con suffissi europei
                for sfx in ['.L', '.DE', '.MI']:
                    try:
                        data = yf.Ticker(t + sfx).history(start=start_date, end=end_date, auto_adjust=True)
                        if not data.empty and len(data) >= 50:
                            break
                    except:
                        continue
            
            if data.empty or len(data) < 50:
                errors.append(t)
                continue
                
            prices = data['Close'].squeeze()
            prices.index = pd.to_datetime(
                prices.index.tz_localize(None) if prices.index.tz else prices.index
            ).normalize()
            all_data[t] = prices
            
        except Exception as e:
            errors.append(t)
    
    if len(all_data) < 2:
        return None, None, errors
    
    df = pd.DataFrame(all_data).ffill(limit=5).bfill(limit=5).dropna()
    prices = df.resample('W').last().dropna()
    returns = prices.pct_change().dropna() * 100
    
    return prices, returns, errors


@st.cache_data(ttl=3600, show_spinner=False)
def download_benchmark_data(ticker, start_date, end_date):
    """Scarica i dati del benchmark con cache."""
    try:
        data = yf.Ticker(ticker).history(start=start_date, end=end_date, auto_adjust=True)
        if data.empty or len(data) < 50:
            return None, None
        
        prices = data['Close'].squeeze()
        prices.index = pd.to_datetime(
            prices.index.tz_localize(None) if prices.index.tz else prices.index
        ).normalize()
        weekly = prices.resample('W').last().dropna()
        
        if isinstance(weekly, pd.DataFrame):
            weekly = weekly.squeeze()
        
        returns = weekly.pct_change().dropna() * 100
        return weekly, returns
    except:
        return None, None


# ════════════════════════════════════════════════════════════════════════════════
# CLASSE BENCHMARK ANALYZER
# ════════════════════════════════════════════════════════════════════════════════

class BenchmarkAnalyzer:
    """Analizza e seleziona automaticamente il benchmark più appropriato."""
    
    def __init__(self, start_date, end_date, risk_free_rate=0.02):
        self.start_date = start_date
        self.end_date = end_date
        self.risk_free_rate = risk_free_rate
        self.benchmark_prices = {}
        self.benchmark_returns = {}
        self.benchmark_metrics = {}
        self.best_benchmark = None

    def download_benchmark_data(self, ticker):
        """Scarica dati benchmark con cache."""
        if ticker in self.benchmark_prices:
            return True
        
        prices, returns = download_benchmark_data(ticker, self.start_date, self.end_date)
        if prices is None:
            return False
        
        self.benchmark_prices[ticker] = prices
        self.benchmark_returns[ticker] = returns
        return len(returns) >= 20

    def calculate_metrics(self, ticker):
        """Calcola le metriche di performance per un benchmark."""
        if not self.download_benchmark_data(ticker):
            return None
        
        ret = self.benchmark_returns[ticker]
        prices = self.benchmark_prices[ticker]
        
        # Rendimento annualizzato
        mu = float(expected_returns.mean_historical_return(
            pd.DataFrame({ticker: prices}), frequency=52
        ).iloc[0]) * 100
        
        rd = ret / 100
        vol = float(rd.std() * np.sqrt(52) * 100)
        excess = mu - self.risk_free_rate * 100
        sharpe = excess / vol if vol > 0 else 0
        
        ds = rd[rd < self.risk_free_rate/52]
        dd = float(np.sqrt(((ds - self.risk_free_rate/52)**2).mean()) * np.sqrt(52) * 100) if len(ds) > 0 else vol * 0.7
        sortino = excess / dd if dd > 0 else 0
        
        cum = (1 + rd).cumprod()
        mdd = float(abs(((cum - cum.expanding().max()) / cum.expanding().max()).min()) * 100)
        
        self.benchmark_metrics[ticker] = {
            'ticker': ticker,
            'mean_return': mu,
            'volatility': vol,
            'sharpe': sharpe,
            'sortino': sortino,
            'max_drawdown': mdd,
            'calmar': mu/mdd if mdd > 0 else 0
        }
        return self.benchmark_metrics[ticker]

    def find_best(self, port_returns, port_metrics):
        """Trova il benchmark più appropriato."""
        results = []
        
        for b in BENCHMARK_CANDIDATES:
            m = self.calculate_metrics(b['ticker'])
            if not m:
                continue
            
            br = self.benchmark_returns[b['ticker']].copy()
            pr = port_returns.copy()
            pr.index = pd.to_datetime(pr.index).normalize()
            br.index = pd.to_datetime(br.index).normalize()
            common = pr.index.intersection(br.index)
            
            if len(common) < 20:
                continue
            
            corr = float(pr.loc[common].corr(br.loc[common]))
            te = float((pr.loc[common] - br.loc[common]).std() * np.sqrt(52))
            
            cov = np.cov(pr.loc[common]/100, br.loc[common]/100)
            beta = cov[0,1]/cov[1,1] if cov[1,1] > 0 else 1.0
            
            score = (corr * 0.5 + 
                    (1/(1+te/10)) * 0.35 + 
                    (1/(1+abs(m['volatility']-port_metrics['vol'])/10)) * 0.15)
            
            results.append({
                **m,
                'name': b['name'],
                'correlation': corr,
                'tracking_error': te,
                'beta': beta,
                'score': score
            })
        
        if not results:
            return None
        
        self.best_benchmark = max(results, key=lambda x: x['score'])
        return self.best_benchmark


# ════════════════════════════════════════════════════════════════════════════════
# CLASSE PORTFOLIO OPTIMIZER
# ════════════════════════════════════════════════════════════════════════════════

class PortfolioOptimizer:
    """Classe principale per l'ottimizzazione del portafoglio."""
    
    def __init__(self, tickers, prices, returns, min_weight=0.01, max_concentration=0.25,
                 risk_free_rate=0.02, sector_map=None, sector_limits=None, 
                 target_volatility=None, start_date=None, end_date=None):
        
        self.tickers = list(prices.columns)
        self.n_assets = len(self.tickers)
        self.prices = prices
        self.returns = returns
        self.min_weight = min_weight
        self.max_concentration = max_concentration
        self.risk_free_rate = risk_free_rate
        self.start_date = start_date
        self.end_date = end_date
        
        self.mu = expected_returns.mean_historical_return(prices, frequency=52)
        self.S = risk_models.sample_cov(prices, frequency=52)
        
        self.sector_map = sector_map if sector_map else DEFAULT_SECTOR_MAP.copy()
        
        if sector_limits is False:
            self.sector_limits = None
            self.use_sector_constraints = False
        elif sector_limits is None:
            self.sector_limits = DEFAULT_SECTOR_LIMITS.copy()
            self.use_sector_constraints = True
        else:
            self.sector_limits = sector_limits
            self.use_sector_constraints = True
        
        self.target_volatility = target_volatility
        self.use_volatility_constraint = target_volatility is not None
        
        self.bench = BenchmarkAnalyzer(start_date, end_date, risk_free_rate)
        self.best_benchmark = None
        self.results = {}

    def _build_sector_mapper(self):
        """Costruisce il mapping ticker->settore."""
        return {ticker: self.sector_map.get(ticker, 'Other') for ticker in self.tickers}

    def _get_active_sectors(self):
        """Ottiene i settori attivi nel portafoglio."""
        sector_mapper = self._build_sector_mapper()
        sectors = {}
        for ticker, sector in sector_mapper.items():
            if sector not in sectors:
                sectors[sector] = []
            sectors[sector].append(ticker)
        return sectors

    def _apply_sector_constraints(self, ef):
        """Applica vincoli settoriali a EfficientFrontier."""
        if not self.use_sector_constraints:
            return ef
        
        sector_mapper = self._build_sector_mapper()
        sector_lower = {}
        sector_upper = {}
        active_sectors = set(sector_mapper.values())
        
        for sector in active_sectors:
            sector_lower[sector] = 0
            sector_upper[sector] = self.sector_limits.get(sector, self.sector_limits.get('Other', 1.0))
        
        try:
            ef.add_sector_constraints(sector_mapper, sector_lower, sector_upper)
        except Exception:
            pass
        
        return ef

    def stats(self, w):
        """Calcola le statistiche del portafoglio."""
        pr = (self.returns * w).sum(axis=1) / 100
        tot = (1 + pr).prod()
        n = len(pr) / 52
        ret = (tot ** (1/n) - 1) * 100 if n > 0 else 0
        vol = float(pr.std() * np.sqrt(52) * 100)
        excess = ret - self.risk_free_rate * 100
        sharpe = excess / vol if vol > 0 else 0
        
        ds = pr[pr < self.risk_free_rate/52]
        dd = float(np.sqrt(((ds - self.risk_free_rate/52)**2).mean()) * np.sqrt(52) * 100) if len(ds) > 0 else vol * 0.7
        sortino = excess / dd if dd > 0 else 0
        
        cum = (1 + pr).cumprod()
        mdd = float(abs(((cum - cum.expanding().max()) / cum.expanding().max()).min()) * 100)
        
        return {
            'ret': ret, 'vol': vol, 'sharpe': sharpe,
            'sortino': sortino, 'mdd': mdd,
            'calmar': ret/mdd if mdd > 0 else 0
        }

    def optimize_max_sharpe(self):
        """Ottimizzazione Max Sharpe."""
        ef = EfficientFrontier(self.mu, self.S, 
                              weight_bounds=(self.min_weight, self.max_concentration))
        
        if self.use_sector_constraints:
            ef = self._apply_sector_constraints(ef)
        
        if self.use_volatility_constraint:
            try:
                ef.efficient_risk(target_volatility=self.target_volatility)
            except:
                ef = EfficientFrontier(self.mu, self.S, 
                                      weight_bounds=(self.min_weight, self.max_concentration))
                if self.use_sector_constraints:
                    ef = self._apply_sector_constraints(ef)
                ef.max_sharpe(risk_free_rate=self.risk_free_rate)
        else:
            ef.max_sharpe(risk_free_rate=self.risk_free_rate)
        
        w = np.array([ef.clean_weights().get(t, 0) for t in self.tickers])
        s = self.stats(w)
        self.results['sharpe'] = {'weights': w, **s}
        return w

    def optimize_max_sortino(self):
        """Ottimizzazione Max Sortino."""
        es = EfficientSemivariance(
            self.mu, self.prices.pct_change().dropna(), frequency=52,
            weight_bounds=(self.min_weight, self.max_concentration)
        )
        
        if self.use_sector_constraints:
            es = self._apply_sector_constraints(es)
        
        if self.use_volatility_constraint:
            try:
                target_semidev = self.target_volatility * 0.8
                es.efficient_risk(target_semideviation=target_semidev)
            except:
                es = EfficientSemivariance(
                    self.mu, self.prices.pct_change().dropna(), frequency=52,
                    weight_bounds=(self.min_weight, self.max_concentration)
                )
                if self.use_sector_constraints:
                    es = self._apply_sector_constraints(es)
                es.max_quadratic_utility(risk_aversion=1)
        else:
            es.max_quadratic_utility(risk_aversion=1)
        
        w = np.array([es.clean_weights().get(t, 0) for t in self.tickers])
        s = self.stats(w)
        self.results['sortino'] = {'weights': w, **s}
        return w

    def optimize_risk_parity(self):
        """Ottimizzazione Risk Parity."""
        cov = self.S.values * 10000
        cov_annual = self.S.values
        
        def obj(w):
            pv = np.sqrt(w.T @ cov @ w)
            if pv < 1e-10:
                return 1e10
            rc = w * (cov @ w) / pv
            return np.sum((rc - pv/self.n_assets)**2)
        
        iv = 1 / np.sqrt(np.diag(cov))
        init = iv / iv.sum()
        
        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
        
        if self.use_sector_constraints:
            active_sectors = self._get_active_sectors()
            for sector, tickers_in_sector in active_sectors.items():
                max_weight = self.sector_limits.get(sector, self.sector_limits.get('Other', 1.0))
                indices = [self.tickers.index(t) for t in tickers_in_sector if t in self.tickers]
                if indices:
                    constraints.append({
                        'type': 'ineq',
                        'fun': lambda w, idx=indices, mw=max_weight: mw - np.sum(w[idx])
                    })
        
        if self.use_volatility_constraint:
            constraints.append({
                'type': 'ineq',
                'fun': lambda w: self.target_volatility**2 - (w.T @ cov_annual @ w)
            })
        
        res = minimize(
            obj, init, method='SLSQP',
            bounds=Bounds([self.min_weight]*self.n_assets, [self.max_concentration]*self.n_assets),
            constraints=constraints
        )
        
        w = res.x
        s = self.stats(w)
        self.results['rp'] = {'weights': w, **s}
        return w

    def optimize_hrp(self):
        """Ottimizzazione HRP."""
        hrp = HRPOpt(self.prices.pct_change().dropna())
        hrp.optimize()
        w = np.array([hrp.clean_weights().get(t, 0) for t in self.tickers])
        
        if self.use_volatility_constraint:
            cov_annual = self.S.values
            port_vol = np.sqrt(w.T @ cov_annual @ w)
            if port_vol > self.target_volatility:
                scale_factor = self.target_volatility / port_vol
                w = w * scale_factor
                w = w / w.sum()
        
        s = self.stats(w)
        self.results['hrp'] = {'weights': w, **s}
        return w

    def find_benchmark(self):
        """Trova il benchmark migliore."""
        eq = np.array([1/self.n_assets] * self.n_assets)
        pr = (self.returns * eq).sum(axis=1)
        s = self.stats(eq)
        self.best_benchmark = self.bench.find_best(pr, {'vol': s['vol'], 'ret': s['ret']})

    def calc_bench_metrics(self):
        """Calcola le metriche relative al benchmark."""
        if not self.best_benchmark:
            return
        
        br = self.bench.benchmark_returns[self.best_benchmark['ticker']].copy()
        br.index = pd.to_datetime(br.index).normalize()
        
        for name, data in self.results.items():
            pr = (self.returns * data['weights']).sum(axis=1)
            pr.index = pd.to_datetime(pr.index).normalize()
            common = pr.index.intersection(br.index)
            
            if len(common) < 20:
                continue
            
            pa, ba = pr.loc[common], br.loc[common]
            cov = np.cov(pa/100, ba/100)
            beta = cov[0,1]/cov[1,1] if cov[1,1] > 0 else 1.0
            te = float((pa - ba).std() * np.sqrt(52))
            n = len(pa)/52
            pc = ((1+pa/100).prod()**(1/n)-1)*100
            bc = ((1+ba/100).prod()**(1/n)-1)*100
            alpha = pc - bc
            ir = alpha/te if te > 0 else 0
            excess_return = data['ret'] - self.risk_free_rate * 100
            treynor = excess_return / beta if beta != 0 else 0
            calmar = data['ret'] / data['mdd'] if data['mdd'] > 0 else 0
            
            self.results[name].update({
                'beta': beta, 'te': te, 'alpha': alpha, 'ir': ir,
                'treynor': treynor, 'calmar': calmar, 'n_weeks': len(common)
            })

    # ════════════════════════════════════════════════════════════════════════════
    # METODI DI PLOTTING (restituiscono oggetti figure)
    # ════════════════════════════════════════════════════════════════════════════

    def plot_frontier(self):
        """Genera il grafico della frontiera efficiente."""
        pts = []
        for t in np.linspace(float(self.mu.min()), float(self.mu.max()), 40):
            try:
                ef = EfficientFrontier(self.mu, self.S, 
                                      weight_bounds=(self.min_weight, self.max_concentration))
                ef.efficient_return(t)
                w = np.array([ef.clean_weights().get(tk, 0) for tk in self.tickers])
                s = self.stats(w)
                pts.append((s['vol'], s['ret'], s['sharpe']))
            except:
                pass
        
        if not pts:
            return None
        
        v, r, sh = zip(*pts)
        fig, ax = plt.subplots(figsize=(12, 8))
        fig.patch.set_facecolor('#0f172a')
        ax.set_facecolor('#1e293b')
        
        sc = ax.scatter(v, r, c=sh, cmap='viridis', s=50, alpha=0.7)
        cbar = plt.colorbar(sc, ax=ax)
        cbar.set_label('Sharpe Ratio', color='white')
        cbar.ax.yaxis.set_tick_params(color='white')
        plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')
        
        colors = {
            'sharpe': ('#2ecc71', '*', 'MAX SHARPE'),
            'sortino': ('#9b59b6', 'P', 'MAX SORTINO'),
            'rp': ('#f39c12', 'D', 'RISK PARITY'),
            'hrp': ('#00bcd4', 'H', 'HRP')
        }
        
        for name, data in self.results.items():
            c, m, label = colors.get(name, ('gray', 'o', name.upper()))
            ax.scatter(data['vol'], data['ret'], c=c, s=300, marker=m,
                      edgecolors='white', linewidth=2, label=label, zorder=10)
        
        ax.set_xlabel('Volatilità Annualizzata (%)', color='white', fontsize=12)
        ax.set_ylabel('Rendimento Annualizzato (%)', color='white', fontsize=12)
        ax.set_title('FRONTIERA EFFICIENTE', color='#00d4ff', fontsize=14, fontweight='bold')
        ax.legend(facecolor='#1e293b', edgecolor='#334155', labelcolor='white')
        ax.tick_params(colors='white')
        ax.grid(True, alpha=0.3, color='#334155')
        
        for spine in ax.spines.values():
            spine.set_color('#334155')
        
        plt.tight_layout()
        return fig

    def plot_cumulative(self):
        """Genera il grafico dei rendimenti cumulativi."""
        fig, ax = plt.subplots(figsize=(12, 7))
        fig.patch.set_facecolor('#0f172a')
        ax.set_facecolor('#1e293b')
        
        norm = self.prices / self.prices.iloc[0] * 100
        colors = {
            'sharpe': '#2ecc71',
            'sortino': '#9b59b6',
            'rp': '#f39c12',
            'hrp': '#00bcd4'
        }
        
        for name, data in self.results.items():
            ax.plot((norm * data['weights']).sum(axis=1),
                   color=colors.get(name, 'gray'), linewidth=2, label=name.upper())
        
        if self.best_benchmark and self.best_benchmark['ticker'] in self.bench.benchmark_prices:
            bp = self.bench.benchmark_prices[self.best_benchmark['ticker']]
            common = norm.index.intersection(bp.index)
            if len(common) > 0:
                ax.plot(bp.loc[common]/bp.loc[common].iloc[0]*100,
                       'w--', linewidth=2, alpha=0.7,
                       label=f"Benchmark: {self.best_benchmark['ticker']}")
        
        ax.axhline(100, color='gray', linestyle=':', alpha=0.5, label='Base 100')
        ax.set_xlabel('Data', color='white', fontsize=12)
        ax.set_ylabel('Valore Portafoglio (Base 100€)', color='white', fontsize=12)
        ax.set_title('RENDIMENTI CUMULATIVI', color='#00d4ff', fontsize=14, fontweight='bold')
        ax.legend(facecolor='#1e293b', edgecolor='#334155', labelcolor='white')
        ax.tick_params(colors='white')
        ax.grid(True, alpha=0.3, color='#334155')
        
        for spine in ax.spines.values():
            spine.set_color('#334155')
        
        plt.tight_layout()
        return fig

    def plot_drawdown(self):
        """Genera il grafico del drawdown."""
        fig, ax = plt.subplots(figsize=(12, 6))
        fig.patch.set_facecolor('#0f172a')
        ax.set_facecolor('#1e293b')
        
        colors = {
            'sharpe': '#2ecc71',
            'sortino': '#9b59b6',
            'rp': '#f39c12',
            'hrp': '#00bcd4'
        }
        
        for name, data in self.results.items():
            pr = (self.returns * data['weights']).sum(axis=1)/100
            cum = (1+pr).cumprod()
            dd = (cum - cum.expanding().max())/cum.expanding().max()*100
            ax.fill_between(dd.index, dd.values, 0, alpha=0.3, color=colors.get(name, 'gray'))
            ax.plot(dd.index, dd.values, color=colors.get(name, 'gray'),
                   label=f"{name.upper()} (Max: {dd.min():.1f}%)")
        
        ax.set_xlabel('Data', color='white', fontsize=12)
        ax.set_ylabel('Drawdown (%)', color='white', fontsize=12)
        ax.set_title('DRAWDOWN NEL TEMPO', color='#00d4ff', fontsize=14, fontweight='bold')
        ax.legend(facecolor='#1e293b', edgecolor='#334155', labelcolor='white')
        ax.tick_params(colors='white')
        ax.grid(True, alpha=0.3, color='#334155')
        
        for spine in ax.spines.values():
            spine.set_color('#334155')
        
        plt.tight_layout()
        return fig

    def plot_correlation_matrix(self):
        """Genera la matrice di correlazione."""
        corr_matrix = self.returns.corr()
        n_assets = len(self.tickers)
        
        fig_width = max(12, n_assets * 0.6)
        fig_height = max(10, n_assets * 0.5)
        
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        fig.patch.set_facecolor('#0f172a')
        ax.set_facecolor('#1e293b')
        
        sns.heatmap(
            corr_matrix,
            annot=True,
            fmt='.2f',
            cmap='RdBu_r',
            center=0,
            vmin=-1, vmax=1,
            square=True,
            linewidths=0.5,
            linecolor='#334155',
            cbar_kws={'label': 'Correlazione', 'shrink': 0.8},
            annot_kws={'size': max(6, 10 - n_assets // 4), 'color': 'white'},
            ax=ax
        )
        
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', color='white')
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, color='white')
        ax.set_title('MATRICE DI CORRELAZIONE', color='#00d4ff', fontsize=14, fontweight='bold', pad=20)
        
        cbar = ax.collections[0].colorbar
        cbar.ax.yaxis.set_tick_params(color='white')
        plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')
        cbar.set_label('Correlazione', color='white')
        
        plt.tight_layout()
        return fig

    def plot_risk_return_map(self):
        """Genera la mappa risk-return."""
        if not self.best_benchmark:
            return None
        
        fig, ax = plt.subplots(figsize=(14, 10))
        fig.patch.set_facecolor('#0f172a')
        ax.set_facecolor('#1e293b')
        
        colors = {
            'sharpe': ('#2ecc71', '*', 'MAX SHARPE'),
            'sortino': ('#9b59b6', 'P', 'MAX SORTINO'),
            'rp': ('#f39c12', 'D', 'RISK PARITY'),
            'hrp': ('#00bcd4', 'H', 'HRP')
        }
        
        # Risk-free point
        ax.scatter(0, self.risk_free_rate * 100, c='gray', s=200, marker='s',
                  edgecolors='white', linewidth=2,
                  label=f'Risk-Free ({self.risk_free_rate*100:.1f}%)', zorder=5)
        
        # Benchmark
        b = self.best_benchmark
        ax.scatter(b['volatility'], b['mean_return'], c='#3b82f6', s=400, marker='X',
                  edgecolors='white', linewidth=2,
                  label=f"BENCHMARK ({b['ticker']})", zorder=10)
        
        # CML line
        if b['sharpe'] > 0:
            vol_range = np.linspace(0, max(b['volatility'] * 1.5, 40), 100)
            cml_returns = self.risk_free_rate * 100 + b['sharpe'] * vol_range
            ax.plot(vol_range, cml_returns, 'w--', alpha=0.3, linewidth=1.5,
                   label=f"CML (Sharpe = {b['sharpe']:.2f})")
        
        # Strategie
        for name, data in self.results.items():
            c, m, label = colors.get(name, ('gray', 'o', name.upper()))
            ax.scatter(data['vol'], data['ret'], c=c, s=350, marker=m,
                      edgecolors='white', linewidth=2, label=label, zorder=10)
        
        ax.set_xlabel('Volatilità Annualizzata (%)', color='white', fontsize=12)
        ax.set_ylabel('Rendimento Annualizzato (%)', color='white', fontsize=12)
        ax.set_title('MAPPA RISK-RETURN', color='#00d4ff', fontsize=14, fontweight='bold')
        ax.legend(facecolor='#1e293b', edgecolor='#334155', labelcolor='white', loc='upper left')
        ax.tick_params(colors='white')
        ax.grid(True, alpha=0.3, color='#334155')
        ax.axhline(0, color='gray', linestyle='-', alpha=0.3)
        
        for spine in ax.spines.values():
            spine.set_color('#334155')
        
        plt.tight_layout()
        return fig

    def plot_allocation_pie(self):
        """Genera i pie chart dell'allocazione."""
        MIN_WEIGHT_THRESHOLD = 0.02
        colors_palette = ['#2ecc71', '#3498db', '#9b59b6', '#f39c12', '#1abc9c',
                         '#e74c3c', '#34495e', '#e67e22', '#27ae60', '#8e44ad',
                         '#2980b9', '#c0392b', '#16a085', '#d35400', '#7f8c8d']
        others_color = '#bdc3c7'
        
        strategy_styles = {
            'sharpe': {'title': 'MAX SHARPE', 'color': '#2ecc71'},
            'sortino': {'title': 'MAX SORTINO', 'color': '#9b59b6'},
            'rp': {'title': 'RISK PARITY', 'color': '#f39c12'},
            'hrp': {'title': 'HRP', 'color': '#00bcd4'}
        }
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 14))
        fig.patch.set_facecolor('#0f172a')
        axes = axes.flatten()
        
        for idx, (name, data) in enumerate(self.results.items()):
            ax = axes[idx]
            ax.set_facecolor('#1e293b')
            
            weights = data['weights']
            sorted_indices = np.argsort(weights)[::-1]
            labels, sizes, colors, explode_list = [], [], [], []
            others_weight = 0
            color_idx = 0
            
            for i in sorted_indices:
                w = weights[i]
                if w < 0.001:
                    continue
                if w < MIN_WEIGHT_THRESHOLD:
                    others_weight += w
                else:
                    labels.append(self.tickers[i])
                    sizes.append(w * 100)
                    colors.append(colors_palette[color_idx % len(colors_palette)])
                    explode_list.append(0.03 if color_idx < 3 else 0)
                    color_idx += 1
            
            if others_weight > 0.001:
                labels.append('Altri')
                sizes.append(others_weight * 100)
                colors.append(others_color)
                explode_list.append(0)
            
            wedges, texts, autotexts = ax.pie(
                sizes, labels=None, colors=colors,
                autopct=lambda pct: f'{pct:.1f}%' if pct >= 3 else '',
                explode=explode_list, startangle=90, pctdistance=0.75,
                wedgeprops={'edgecolor': 'white', 'linewidth': 2},
                textprops={'fontsize': 9, 'fontweight': 'bold', 'color': 'white'}
            )
            
            for autotext in autotexts:
                autotext.set_color('white')
            
            centre_circle = plt.Circle((0, 0), 0.4, fc='#1e293b', ec='#334155', linewidth=2)
            ax.add_patch(centre_circle)
            
            style = strategy_styles.get(name, {'title': name.upper(), 'color': 'gray'})
            center_text = f"Sharpe\n{data['sharpe']:.2f}"
            ax.text(0, 0, center_text, ha='center', va='center',
                   fontsize=12, fontweight='bold', color='white')
            
            ax.set_title(style['title'], fontsize=14, fontweight='bold',
                        color=style['color'], pad=10)
            
            legend_labels = [f"{l} ({s:.1f}%)" for l, s in zip(labels, sizes)]
            leg = ax.legend(wedges, legend_labels, title="Asset", loc='center left',
                           bbox_to_anchor=(1, 0, 0.5, 1), fontsize=8, title_fontsize=9)
            leg.get_frame().set_facecolor('#1e293b')
            leg.get_frame().set_edgecolor('#334155')
            for text in leg.get_texts():
                text.set_color('white')
            leg.get_title().set_color('white')
        
        fig.suptitle('ALLOCAZIONE PORTAFOGLIO', fontsize=16, fontweight='bold',
                    color='#00d4ff', y=0.98)
        
        plt.tight_layout(rect=[0, 0.02, 1, 0.95])
        return fig

    def run_full_optimization(self, progress_callback=None):
        """Esegue l'ottimizzazione completa."""
        steps = [
            ("Ottimizzazione Max Sharpe...", self.optimize_max_sharpe),
            ("Ottimizzazione Max Sortino...", self.optimize_max_sortino),
            ("Ottimizzazione Risk Parity...", self.optimize_risk_parity),
            ("Ottimizzazione HRP...", self.optimize_hrp),
            ("Ricerca benchmark...", self.find_benchmark),
            ("Calcolo metriche benchmark...", self.calc_bench_metrics),
        ]
        
        for i, (msg, func) in enumerate(steps):
            if progress_callback:
                progress_callback((i + 1) / len(steps), msg)
            func()
        
        return self.results


# ════════════════════════════════════════════════════════════════════════════════
# INTERFACCIA STREAMLIT
# ════════════════════════════════════════════════════════════════════════════════

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>📊 Portfolio Optimizer Pro</h1>
        <p>Sistema avanzato di ottimizzazione del portafoglio con 4 strategie quantitative</p>
    </div>
    """, unsafe_allow_html=True)
    
    # ════════════════════════════════════════════════════════════════════════════
    # SIDEBAR - PANNELLO DI CONTROLLO
    # ════════════════════════════════════════════════════════════════════════════
    
    with st.sidebar:
        st.markdown("## ⚙️ Pannello di Controllo")
        st.markdown("---")
        
        # Input Tickers
        st.markdown("### 📈 ETF da Analizzare")
        tickers_input = st.text_area(
            "Inserisci i ticker (uno per riga)",
            value=DEFAULT_TICKERS,
            height=200,
            help="Inserisci i simboli degli ETF, uno per riga. Es: SPY, QQQ, VTI"
        )
        
        st.markdown("---")
        
        # Parametri Temporali
        st.markdown("### 📅 Periodo di Analisi")
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input(
                "Data Inizio",
                value=datetime(2022, 1, 1),
                min_value=datetime(2010, 1, 1),
                max_value=datetime.today() - timedelta(days=365)
            )
        with col2:
            end_date = st.date_input(
                "Data Fine",
                value=datetime.today(),
                min_value=start_date + timedelta(days=365),
                max_value=datetime.today()
            )
        
        st.markdown("---")
        
        # Parametri Tecnici
        st.markdown("### 🔧 Parametri Tecnici")
        
        risk_free_rate = st.number_input(
            "Risk-Free Rate (%)",
            min_value=0.0,
            max_value=10.0,
            value=3.7,
            step=0.1,
            help="Tasso risk-free annuale (es. rendimento BOT o Treasury)"
        ) / 100
        
        col1, col2 = st.columns(2)
        with col1:
            min_weight = st.number_input(
                "Peso Min (%)",
                min_value=0.0,
                max_value=10.0,
                value=1.0,
                step=0.5
            ) / 100
        with col2:
            max_weight = st.number_input(
                "Peso Max (%)",
                min_value=10.0,
                max_value=100.0,
                value=25.0,
                step=5.0
            ) / 100
        
        st.markdown("---")
        
        # Vincoli Settoriali
        st.markdown("### 🏭 Vincoli Settoriali")
        use_sector_constraints = st.checkbox(
            "Attiva Vincoli Settoriali",
            value=True,
            help="Limita l'esposizione massima per settore"
        )
        
        sector_limits = None
        if use_sector_constraints:
            with st.expander("📋 Configura Limiti Settoriali", expanded=False):
                st.markdown("*Limiti massimi per settore (%)*")
                sector_limits = {}
                cols = st.columns(2)
                for i, (sector, default_limit) in enumerate(DEFAULT_SECTOR_LIMITS.items()):
                    with cols[i % 2]:
                        sector_limits[sector] = st.slider(
                            sector,
                            min_value=10,
                            max_value=50,
                            value=int(default_limit * 100),
                            step=5
                        ) / 100
        else:
            sector_limits = False
        
        st.markdown("---")
        
        # Vincolo Volatilità
        st.markdown("### 📊 Vincolo Volatilità")
        use_vol_constraint = st.checkbox(
            "Attiva Target Volatilità",
            value=False,
            help="Limita la volatilità massima del portafoglio"
        )
        
        target_volatility = None
        if use_vol_constraint:
            target_volatility = st.slider(
                "Volatilità Target (%)",
                min_value=10,
                max_value=40,
                value=22,
                step=1
            ) / 100
        
        st.markdown("---")
        
        # Pulsante Analisi
        run_analysis = st.button(
            "🚀 Lancia Analisi",
            type="primary",
            use_container_width=True
        )
    
    # ════════════════════════════════════════════════════════════════════════════
    # AREA PRINCIPALE
    # ════════════════════════════════════════════════════════════════════════════
    
    if run_analysis:
        # Parse tickers
        tickers = [t.strip().upper() for t in tickers_input.split('\n') if t.strip()]
        
        if len(tickers) < 2:
            st.error("⚠️ Inserisci almeno 2 ticker per l'analisi!")
            return
        
        # Progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Download dati
        status_text.text("📥 Download dati da Yahoo Finance...")
        progress_bar.progress(10)
        
        prices, returns, errors = download_data(
            tickers,
            start_date.strftime('%Y-%m-%d'),
            end_date.strftime('%Y-%m-%d')
        )
        
        if prices is None:
            st.error("❌ Impossibile scaricare dati sufficienti. Verifica i ticker inseriti.")
            return
        
        if errors:
            st.warning(f"⚠️ Ticker non trovati o con dati insufficienti: {', '.join(errors)}")
        
        progress_bar.progress(30)
        status_text.text("🔄 Inizializzazione ottimizzatore...")
        
        # Crea ottimizzatore
        optimizer = PortfolioOptimizer(
            tickers=tickers,
            prices=prices,
            returns=returns,
            min_weight=min_weight,
            max_concentration=max_weight,
            risk_free_rate=risk_free_rate,
            sector_limits=sector_limits,
            target_volatility=target_volatility,
            start_date=start_date.strftime('%Y-%m-%d'),
            end_date=end_date.strftime('%Y-%m-%d')
        )
        
        # Callback per progress
        def update_progress(pct, msg):
            progress_bar.progress(int(30 + pct * 60))
            status_text.text(f"🔄 {msg}")
        
        # Esegui ottimizzazione
        results = optimizer.run_full_optimization(progress_callback=update_progress)
        
        progress_bar.progress(100)
        status_text.text("✅ Analisi completata!")
        
        # Store in session state
        st.session_state['optimizer'] = optimizer
        st.session_state['results'] = results
        st.session_state['analysis_done'] = True
    
    # ════════════════════════════════════════════════════════════════════════════
    # VISUALIZZAZIONE RISULTATI
    # ════════════════════════════════════════════════════════════════════════════
    
    if st.session_state.get('analysis_done', False):
        optimizer = st.session_state['optimizer']
        results = st.session_state['results']
        
        # Tabs
        tab1, tab2, tab3, tab4 = st.tabs([
            "📊 Dashboard & Metriche",
            "🎯 Allocazione",
            "📈 Analisi Grafica",
            "📚 Glossario"
        ])
        
        # ════════════════════════════════════════════════════════════════════════
        # TAB 1: DASHBOARD & METRICHE
        # ════════════════════════════════════════════════════════════════════════
        
        with tab1:
            st.markdown("### 🎯 Benchmark Selezionato")
            
            if optimizer.best_benchmark:
                b = optimizer.best_benchmark
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value">{b['ticker']}</div>
                        <div class="metric-label">{b['name']}</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value">{b['correlation']:.3f}</div>
                        <div class="metric-label">Correlazione</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value">{b['score']:.4f}</div>
                        <div class="metric-label">Score</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col4:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value">{b['sharpe']:.2f}</div>
                        <div class="metric-label">Sharpe Bench</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.info(f"💡 **Motivazione:** Il benchmark {b['ticker']} ({b['name']}) è stato selezionato perché presenta la correlazione più alta ({b['correlation']:.3f}) con il portafoglio, un tracking error contenuto ({b['tracking_error']:.2f}%) e una volatilità simile.")
            
            st.markdown("---")
            st.markdown("### 📊 Confronto Strategie")
            
            # Trova la strategia migliore
            best_strategy = max(results.items(), key=lambda x: x[1]['sharpe'])
            best_name, best_data = best_strategy
            
            # KPI della strategia migliore
            st.markdown(f"#### 🏆 Strategia Migliore: **{best_name.upper()}**")
            
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                st.metric("Sharpe Ratio", f"{best_data['sharpe']:.3f}")
            with col2:
                st.metric("Rendimento", f"{best_data['ret']:.2f}%")
            with col3:
                st.metric("Volatilità", f"{best_data['vol']:.2f}%")
            with col4:
                st.metric("Max Drawdown", f"{best_data['mdd']:.2f}%")
            with col5:
                st.metric("Sortino", f"{best_data['sortino']:.3f}")
            
            st.markdown("---")
            
            # DataFrame comparativo
            st.markdown("#### 📋 Tabella Comparativa")
            
            metrics_data = []
            for name, data in results.items():
                row = {
                    'Strategia': name.upper(),
                    'Rendimento (%)': f"{data['ret']:.2f}",
                    'Volatilità (%)': f"{data['vol']:.2f}",
                    'Sharpe': f"{data['sharpe']:.3f}",
                    'Sortino': f"{data['sortino']:.3f}",
                    'Max DD (%)': f"{data['mdd']:.2f}",
                    'Calmar': f"{data.get('calmar', 0):.3f}",
                }
                if 'beta' in data:
                    row.update({
                        'Beta': f"{data['beta']:.3f}",
                        'Alpha (%)': f"{data['alpha']:+.2f}",
                        'IR': f"{data['ir']:.3f}",
                        'Treynor': f"{data.get('treynor', 0):.3f}",
                    })
                metrics_data.append(row)
            
            # Aggiungi benchmark
            if optimizer.best_benchmark:
                b = optimizer.best_benchmark
                bench_row = {
                    'Strategia': f"📌 {b['ticker']}",
                    'Rendimento (%)': f"{b['mean_return']:.2f}",
                    'Volatilità (%)': f"{b['volatility']:.2f}",
                    'Sharpe': f"{b['sharpe']:.3f}",
                    'Sortino': f"{b['sortino']:.3f}",
                    'Max DD (%)': f"{b['max_drawdown']:.2f}",
                    'Calmar': f"{b['calmar']:.3f}",
                }
                if 'beta' in results[list(results.keys())[0]]:
                    bench_row.update({
                        'Beta': '1.000',
                        'Alpha (%)': '0.00',
                        'IR': '0.000',
                        'Treynor': f"{(b['mean_return'] - optimizer.risk_free_rate*100):.3f}",
                    })
                metrics_data.append(bench_row)
            
            df_metrics = pd.DataFrame(metrics_data)
            
            # Styling del dataframe
            def highlight_best(s):
                if s.name == 'Strategia':
                    return [''] * len(s)
                
                # Converti a numerico per confronto
                numeric_vals = pd.to_numeric(s.str.replace('%', '').str.replace('+', ''), errors='coerce')
                
                if s.name in ['Volatilità (%)', 'Max DD (%)']:
                    # Per queste metriche, il minimo è migliore
                    best_idx = numeric_vals.idxmin() if numeric_vals.notna().any() else None
                else:
                    # Per le altre, il massimo è migliore
                    best_idx = numeric_vals.idxmax() if numeric_vals.notna().any() else None
                
                return ['background-color: rgba(46, 204, 113, 0.3)' if i == best_idx else '' for i in range(len(s))]
            
            st.dataframe(
                df_metrics.style.apply(highlight_best),
                use_container_width=True,
                hide_index=True
            )
        
        # ════════════════════════════════════════════════════════════════════════
        # TAB 2: ALLOCAZIONE
        # ════════════════════════════════════════════════════════════════════════
        
        with tab2:
            st.markdown("### 🥧 Allocazione del Portafoglio")
            
            # Pie charts
            fig_pie = optimizer.plot_allocation_pie()
            st.pyplot(fig_pie)
            plt.close(fig_pie)
            
            st.markdown("---")
            st.markdown("### 📋 Pesi Esatti per Strategia")
            
            # Tabella pesi
            weights_data = {'ETF': optimizer.tickers}
            for name, data in results.items():
                weights_data[name.upper()] = [f"{w*100:.2f}%" for w in data['weights']]
            
            df_weights = pd.DataFrame(weights_data)
            st.dataframe(df_weights, use_container_width=True, hide_index=True)
            
            # Download CSV
            csv = df_weights.to_csv(index=False)
            st.download_button(
                label="📥 Scarica Allocazioni (CSV)",
                data=csv,
                file_name="portfolio_allocation.csv",
                mime="text/csv"
            )
        
        # ════════════════════════════════════════════════════════════════════════
        # TAB 3: ANALISI GRAFICA
        # ════════════════════════════════════════════════════════════════════════
        
        with tab3:
            st.markdown("### 📈 Grafici di Analisi")
            
            # Frontiera Efficiente
            st.markdown("#### Frontiera Efficiente")
            fig_frontier = optimizer.plot_frontier()
            if fig_frontier:
                st.pyplot(fig_frontier)
                plt.close(fig_frontier)
            
            st.markdown("---")
            
            # Rendimenti Cumulativi
            st.markdown("#### Rendimenti Cumulativi")
            fig_cumulative = optimizer.plot_cumulative()
            st.pyplot(fig_cumulative)
            plt.close(fig_cumulative)
            
            st.markdown("---")
            
            # Drawdown
            st.markdown("#### Drawdown")
            fig_drawdown = optimizer.plot_drawdown()
            st.pyplot(fig_drawdown)
            plt.close(fig_drawdown)
            
            st.markdown("---")
            
            # Risk-Return Map
            st.markdown("#### Mappa Risk-Return")
            fig_rr = optimizer.plot_risk_return_map()
            if fig_rr:
                st.pyplot(fig_rr)
                plt.close(fig_rr)
            
            st.markdown("---")
            
            # Matrice di Correlazione
            st.markdown("#### Matrice di Correlazione")
            fig_corr = optimizer.plot_correlation_matrix()
            st.pyplot(fig_corr)
            plt.close(fig_corr)
        
        # ════════════════════════════════════════════════════════════════════════
        # TAB 4: GLOSSARIO
        # ════════════════════════════════════════════════════════════════════════
        
        with tab4:
            st.markdown(GLOSSARIO)
    
    else:
        # Stato iniziale - mostra istruzioni
        st.markdown("""
        <div style="text-align: center; padding: 3rem;">
            <h2 style="color: #94a3b8;">👈 Configura i parametri nella sidebar</h2>
            <p style="color: #64748b; font-size: 1.1rem;">
                Inserisci i ticker degli ETF, seleziona il periodo di analisi e 
                clicca su <strong>"Lancia Analisi"</strong> per iniziare.
            </p>
            <br>
            <p style="color: #475569;">
                💡 <strong>Suggerimento:</strong> I ticker di default sono ETF tecnologici europei. 
                Puoi sostituirli con qualsiasi ETF disponibile su Yahoo Finance.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Feature highlights
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("""
            <div class="metric-card">
                <div class="metric-value">4</div>
                <div class="metric-label">Strategie</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="metric-card">
                <div class="metric-value">13+</div>
                <div class="metric-label">Metriche</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="metric-card">
                <div class="metric-value">5</div>
                <div class="metric-label">Grafici</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown("""
            <div class="metric-card">
                <div class="metric-value">Auto</div>
                <div class="metric-label">Benchmark</div>
            </div>
            """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ════════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    main()
