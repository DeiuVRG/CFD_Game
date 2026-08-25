# PROMPT: JOC DE TRADING COMPETITIV PENTRU DESCOPERIREA PONDERILOR OPTIME ALE INDICATORILOR TEHNICI

## OBIECTIV PRINCIPAL

Creează un sistem de simulare competitivă între maxim 10 participanți (procese/agenți AI autonomi) care dezvoltă și testează strategii de trading bazate pe indicatori tehnici, date fundamentale și sentiment analysis. Scopul este identificarea ponderilor optime pentru indicatori tehnici care sunt UTILE ÎN REALITATE, nu doar pe date istorice.

---

## PARTEA I: CONFIGURARE GENERALĂ

### 1.1 Participanți

- **Număr**: 2-10 jucători (procese autonome)
- **Rol**: Fiecare jucător este un trader autonom care dezvoltă propria strategie
- **Capital inițial**: $100,000 pentru fiecare jucător
- **Libertate totală**: Fiecare jucător este COMPLET LIBER să aleagă orice metodă, combinație de metode, sau abordare proprie pentru determinarea ponderilor. Nu există restricții de unicitate - mai mulți jucători pot folosi aceeași metodă sau metode similare.
- **Resurse computaționale**: Fiecare jucător poate utiliza GPU atunci când este disponibil pentru accelerarea calculelor (training de modele ML, optimizări complexe, simulări Monte Carlo, etc.). Implementările trebuie să fie adaptabile - să funcționeze pe CPU dacă GPU nu este disponibil.

### 1.2 Date Disponibile

**Structura în trei segmente (CRUCIAL pentru utilitate reală):**

```
TRAINING SET (60% din date): 2015-2019
- Jucătorii dezvoltă și calibrează ponderile
- Experimentare liberă
- Acces complet pentru optimizare

VALIDATION SET (20% din date): 2020-2021
- Competiția propriu-zisă
- Jocul se desfășoară aici
- Eliminări și clasament

TEST SET (20% din date): 2022-2023
- NECUNOSCUT jucătorilor până la final
- Evaluare finală out-of-sample
- Determină câștigătorul REAL
```

**Conținut date furnizate de Crupier:**
- Prețuri istorice S&P 500 (toate companiile)
- OHLC (Open, High, Low, Close)
- Volume
- Date/timestamp
- Indicatori tehnici calculați la cerere

**Date suplimentare (OPȚIONAL - procurate de jucători):**

Jucătorii pot îmbogăți strategiile cu date externe pe care le procură SINGURI:

**Surse permise:**
- Știri financiare (news APIs, RSS feeds, scrapers)
- Rapoarte financiare (SEC filings, quarterly reports, 10-K, 10-Q)
- Social media sentiment (Twitter/X, Reddit, StockTwits)
- Economic indicators (GDP, unemployment, interest rates)
- Sector/industry reports
- Analyst ratings și price targets
- Earnings call transcripts
- Orice alte surse publice de informații

**PROCES OBLIGATORIU pentru date externe:**

```python
class ProcurareDate:
    def solicita_aprobare_sursa(self, jucator, descriere_sursa):
        """
        Înainte de a folosi orice date externe, jucătorul TREBUIE:
        1. Să trimită către Crupier descrierea sursei
        2. Să primească aprobare
        3. Să trimită datele procurate către Crupier pentru transparență
        """
        cerere = {
            'jucator_id': jucator.id,
            'tip_sursa': 'news' | 'financial_reports' | 'sentiment' | 'economic' | 'other',
            'descriere': 'Ex: Yahoo Finance News API pentru articole despre companiile din portofoliu',
            'frecventa_actualizare': 'zilnic' | 'saptamanal' | 'lunar' | 'la_cerere',
            'exemplu_date': {...}  # Sample pentru review
        }
        
        # Jucător → Crupier: Cerere aprobare
        aprobare = crupier.aproba_sursa_externa(cerere)
        
        if aprobare['status'] == 'APPROVED':
            return True
        else:
            return False  # Nu poate folosi sursa
    
    def trimite_date_procurate(self, jucator, sursa_id, date):
        """
        După procurare, jucătorul trimite datele către Crupier
        pentru audit și transparență
        """
        pachet = {
            'jucator_id': jucator.id,
            'sursa_id': sursa_id,
            'timestamp_procurare': datetime.now(),
            'date': date,  # Datele efective (JSON, text, structured)
            'hash_verificare': hashlib.sha256(str(date).encode()).hexdigest()
        }
        
        # Jucător → Crupier: Raportare date procurate
        confirmare = crupier.inregistreaza_date_externe(pachet)
        
        return confirmare
```

**Reguli pentru date externe:**

1. **Transparență totală**: 
   - Toate sursele trebuie declarate și aprobate de Crupier
   - Toate datele procurate trebuie trimise către Crupier
   - Crupierul menține un registru complet pentru audit

2. **Disponibilitate echitabilă**:
   - Doar surse de date PUBLICE (nu proprietare, nu plătite)
   - Orice jucător poate accesa aceleași surse
   - Avantajul competitiv vine din PROCESAREA datelor, nu din accesul exclusiv

3. **Integritate temporală**:
   - Nu se permit date "din viitor" (look-ahead bias)
   - Datele folosite pentru o decizie la data X trebuie să fie disponibile ÎN REALITATE la data X
   - Crupierul verifică timestamp-urile

4. **Limite de volum**:
   - Maximum 10 surse externe diferite per jucător
   - Maximum 1GB date externe total per jucător
   - Limitări pentru a preveni spam sau abuz

**Exemple de utilizare:**

```python
# Exemplu 1: News sentiment pentru AAPL
class JucatorCuNews:
    def setup_news_source(self):
        # 1. Cere aprobare
        approved = self.solicita_aprobare_sursa(
            tip_sursa='news',
            descriere='NewsAPI.org - articole despre companiile din S&P500',
            frecventa='zilnic'
        )
        
        if approved:
            # 2. Procură știri
            news_data = self.fetch_news_from_api('AAPL', date='2020-01-15')
            
            # 3. Trimite către Crupier
            self.trimite_date_procurate('news_api', news_data)
            
            # 4. Procesează pentru sentiment
            sentiment_score = self.analyze_sentiment(news_data)
            
            # 5. Ajustează ponderi bazat pe sentiment
            adjusted_weights = self.adjust_weights_with_sentiment(
                self.ponderi, 
                sentiment_score
            )
            
            return adjusted_weights

# Exemplu 2: Earnings reports pentru decizie
class JucatorCuFundamentals:
    def incorporate_earnings(self, companie, data):
        # 1. Procură raport earnings
        earnings = self.fetch_sec_filing(companie, '10-Q', data)
        
        # 2. Raportează către Crupier
        self.trimite_date_procurate('sec_filings', earnings)
        
        # 3. Extrage metrici cheie
        eps = self.extract_eps(earnings)
        revenue_growth = self.extract_revenue_growth(earnings)
        
        # 4. Combină cu indicatori tehnici
        technical_signal = self.calculate_technical_signal(companie, data)
        fundamental_signal = self.calculate_fundamental_signal(eps, revenue_growth)
        
        # 5. Decizie finală: 70% tehnic, 30% fundamental
        final_signal = 0.7 * technical_signal + 0.3 * fundamental_signal
        
        return final_signal
```

### 1.3 Rol Crupier (Sistem Central)

**Responsabilități:**
1. **Furnizare date**: Calculează și furnizează indicatorii tehnici solicitați
2. **Execuție tranzacții**: Primește, validează și execută TOATE tranzacțiile jucătorilor
3. **Aplicare costuri**: Calculează și aplică costuri de tranzacționare realiste (comisioane, spread, slippage)
4. **Gestiune portofolii**: Actualizează automat portofoliile tuturor jucătorilor după fiecare tranzacție
5. **Monitorizare**: Urmărește performanța cu metrici avansate
6. **Eliminare**: Aplică regulile de eliminare conform criteriilor stabilite
7. **Recalibrări**: Gestionează punctele de recalibrare periodică
8. **Validare statistică**: Efectuează validare statistică finală
9. **Audit complet**: Menține istoric detaliat al tuturor tranzacțiilor pentru transparență
10. **Gestiune date externe**: 
    - Primește și aprobă cereri pentru surse de date externe
    - Înregistrează toate datele procurate de jucători (știri, rapoarte financiare, etc.)
    - Verifică integritatea temporală (no look-ahead bias)
    - Menține registru transparent pentru audit
11. **Meta-analiză**: Colectează și analizează metodele folosite (pentru învățare meta-nivel)

---

## PARTEA II: FAZA DE PREGĂTIRE

### 2.1 Procesul de Înregistrare

**Pentru fiecare jucător:**

**STEP 1 - Declararea Metodei (OPȚIONAL):**
```
Jucător → Crupier: "Metoda/metodele mele sunt: [DESCRIERE]"

Fiecare jucător poate folosi:
- O singură metodă simplă
- Combinații de metode (ensemble, hybrid)
- Metode adaptative care evoluează în timp
- Orice altă abordare creativă

Exemple de abordări (NON-EXHAUSTIV, doar pentru inspirație):

METODE SIMPLE:
- Machine Learning: Random Forest, Neural Networks, XGBoost, SVM
- Optimizare: Algoritmi genetici, PSO, Simulated Annealing, Bayesian Optimization
- Statistice: Corelație, PCA, Analiza covarianței, Information Theory
- Euristice: Reguli bazate pe experiență, Weighted voting
- Probabilistice: Bayesian Inference, Monte Carlo

COMBINAȚII ȘI HIBRIDE:
- Ensemble: Voting între mai multe metode (ex: RF + XGBoost + SVM)
- Stacking: Meta-model care învață să combine predicțiile
- Boosting: Combinație secvențială de modele slabe
- Cascade: Metode aplicate în lanț (ex: PCA → SVM → Bayesian fine-tuning)
- Adaptive: Schimbă metoda în funcție de regimul pieței

ABORDĂRI AVANSATE:
- Multi-objective optimization (Pareto frontier)
- Reinforcement Learning cu continuous adaptation
- Transfer Learning de la alte piețe/perioade
- Attention mechanisms pentru ponderi dinamice
- Meta-learning: învață ce metodă funcționează când
- Deep Learning: CNNs pentru pattern recognition, LSTMs pentru time series
- Gradient-based optimization pe GPU pentru viteza maximă
- Simulări masive paralele (Monte Carlo, genetic algorithms) pe GPU

UTILIZAREA GPU:
- Implementările pot detecta și utiliza GPU când este disponibil
- Framework-uri recomandate: PyTorch, TensorFlow, JAX, CuPy, Rapids
- Exemple de utilizare:
  * Training accelerat de rețele neurale
  * Optimizare paralela de hiperparametri
  * Backtesting paralel pe mii de configurații
  * Simulări Monte Carlo masive
  * Matrix operations la scară mare
- Fallback automat pe CPU dacă GPU nu este disponibil
- Benchmarking: timpul de calcul NU este limitat, dar strategiile mai rapide pot experimenta mai mult

NOTA: Descrierea metodei este OPȚIONALĂ și servește doar scopuri de documentare.
Crupierul NU validează sau restricționează metodele - orice abordare este permisă.
```

**STEP 2 - Solicitarea Indicatorilor:**
```
Jucător → Crupier: "Indicatorii tehnici pe care îi solicit:"
[LISTĂ DE INDICATORI]

Indicatori disponibili:
TREND:
- SMA (Simple Moving Average) - parametri: perioada
- EMA (Exponential Moving Average) - parametri: perioada
- MACD (Moving Average Convergence Divergence) - parametri: fast, slow, signal
- ADX (Average Directional Index) - parametri: perioada
- Parabolic SAR - parametri: acceleration factor
- Aroon Indicator - parametri: perioada

MOMENTUM:
- RSI (Relative Strength Index) - parametri: perioada
- Stochastic Oscillator - parametri: %K period, %D period
- Williams %R - parametri: perioada
- ROC (Rate of Change) - parametri: perioada
- CCI (Commodity Channel Index) - parametri: perioada
- MFI (Money Flow Index) - parametri: perioada

VOLATILITATE:
- Bollinger Bands - parametri: perioada, deviații standard
- ATR (Average True Range) - parametri: perioada
- Standard Deviation - parametri: perioada
- Keltner Channels - parametri: perioada, multiplier

VOLUM:
- OBV (On-Balance Volume)
- Volume SMA - parametri: perioada
- VWAP (Volume Weighted Average Price)
- Chaikin Money Flow - parametri: perioada

CUSTOM:
- Jucătorii pot solicita combinații sau calcule specifice
```

**STEP 3 - Primirea Datelor:**
```
Crupier → Jucător: Returnează toate indicatorii calculați pentru TRAINING SET
Format: DataFrame cu coloane [data, companie, indicator1, indicator2, ...]
```

**STEP 4 - Calcularea Ponderilor:**
```
Jucător aplică metoda/metodele sale pe TRAINING SET și determină:

ponderi = {
    'RSI': 0.25,
    'MACD': 0.30,
    'Bollinger_Bands': 0.20,
    'Volume_SMA': 0.15,
    'ADX': 0.10
}

Reguli pentru ponderi:
- Suma ponderilor = 1.0 (normalizat)
- Fiecare pondere ∈ [0, 1]
- Minim 3 indicatori, maxim 20 indicatori

LIBERTATE TOTALĂ în determinarea ponderilor:
- Ponderi statice (fixe pe toată perioada)
- Ponderi dinamice (se schimbă în funcție de condiții)
- Ponderi adaptative (învață din performanță)
- Ponderi context-dependente (diferite per companie/sector/regim)
- Orice altă abordare creativă

ACCELERARE GPU (când disponibil):
- Training de modele deep learning pentru predicția ponderilor optime
- Grid search masiv paralel pentru hiperparametri
- Backtesting paralel pe toate combinațiile de ponderi
- Optimizări evolutionary paralele (genetic algorithms la scară mare)
- Simulări Monte Carlo pentru validarea robustității

Exemplu cod pentru detecție GPU:
```python
import torch

class JucatorCuGPU:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Folosesc: {self.device}")
    
    def optimize_weights_gpu(self, training_data):
        # Mută datele pe GPU dacă este disponibil
        data_tensor = torch.tensor(training_data).to(self.device)
        # ... procesare accelerată ...
        return optimized_weights
```
```

---

## PARTEA III: REGULI DE JOC (VALIDATION SET)

### 3.1 Parametri Temporali

```
Durata jocului: 24 luni (perioadă de validare)
Frecvență evaluare: Lunară
Recalibrare permisă: Trimestrial (max 8 recalibrări)

Timeline exemplu:
Luna 1-3: Perioada 1 → Evaluare → Eventuală eliminare
Luna 4-6: Perioada 2 → Recalibrare permisă → Evaluare → Eventuală eliminare
Luna 7-9: Perioada 3 → Evaluare → Eventuală eliminare
Luna 10-12: Perioada 4 → Recalibrare permisă → Evaluare → Eventuală eliminare
... și așa mai departe
```

### 3.2 Reguli de Tranzacționare

**Frecvență obligatorie:**
- MINIMUM 1 tranzacție per lună
- MAXIMUM 50 tranzacții per lună (anti-overtrading)

**Proces de tranzacționare:**

```python
# TOATE tranzacțiile se execută PRIN Crupier
class ProcessTranzactionare:
    def executa_tranzactie(self, jucator, tranzactie):
        """
        Jucător → Crupier: Solicitare tranzacție
        Crupier → Jucător: Confirmare execuție + costuri
        """
        # 1. Jucătorul trimite ordinul către Crupier
        ordine = {
            'jucator_id': jucator.id,
            'actiune': 'BUY' | 'SELL',
            'companie': 'AAPL',
            'cantitate': 100,
            'timestamp': datetime.now()
        }
        
        # 2. Crupierul validează ordinul
        validare = self.valideaza_ordine(ordine, jucator)
        if not validare['valid']:
            return {'status': 'REJECTED', 'motiv': validare['motiv']}
        
        # 3. Crupierul calculează costurile
        costuri = CosturiRealiste().cost_tranzactie(
            ordine['cantitate'] * pret_curent,
            ordine['cantitate'],
            volum_mediu_zilnic
        )
        
        # 4. Crupierul aplică constrângeri de lichiditate
        executie = ConstrangeriLichiditate().limiteaza_ordine(
            ordine['cantitate'],
            volum_mediu_zilnic
        )
        
        # 5. Crupierul actualizează portofoliul jucătorului
        self.actualizeaza_portofoliu(jucator, executie, costuri)
        
        # 6. Crupierul înregistrează tranzacția în istoric
        self.log_tranzactie(ordine, executie, costuri)
        
        # 7. Crupierul returnează confirmarea
        return {
            'status': 'EXECUTED',
            'cantitate_executata': executie['cantitate_executata'],
            'pret_executie': executie['pret_executie'],
            'costuri_totale': costuri,
            'nou_cash': jucator.portofoliu['cash'],
            'noua_pozitie': jucator.portofoliu['pozitii'][ordine['companie']]
        }
```

**Responsabilități Crupier în tranzacționare:**
1. **Primește** toate solicitările de tranzacții de la jucători
2. **Validează** fiecare ordine (capital suficient, limite, reguli)
3. **Calculează** și aplică costuri realiste (comisioane, spread, slippage)
4. **Execută** ordinele la prețul de piață curent
5. **Aplică** constrângeri de lichiditate (limite de volum)
6. **Actualizează** automat portofoliile jucătorilor
7. **Înregistrează** toate tranzacțiile pentru audit și analiză
8. **Raportează** confirmări și erori către jucători

**Avantajele centralizării prin Crupier:**
- Transparență totală - toate tranzacțiile sunt auditate
- Aplicare consistentă a regulilor și costurilor
- Prevenirea fraudei sau manipulării
- Sincronizare corectă a prețurilor de piață
- Istoric complet pentru analiză post-joc

**Logica de decizie (rămâne la jucător):**

```python
# Fiecare jucător implementează propria logică
def decide_tranzactie(self, companie, data_curenta):
    # Obține valorile indicatorilor pentru compania respectivă
    indicatori_valori = self.get_indicator_values(companie, data_curenta)
    
    # Calculează scor agregat folosind ponderile
    scor = 0
    for indicator, valoare in indicatori_valori.items():
        semnal = self.interpret_indicator(indicator, valoare)  # -1 to +1
        scor += self.ponderi[indicator] * semnal
    
    # Decizie finală
    if scor > 0.3:
        actiune = "BUY"
        cantitate = self.calculate_quantity(scor)
    elif scor < -0.3:
        actiune = "SELL"
        cantitate = self.calculate_quantity(abs(scor))
    else:
        return None  # Nu face nimic
    
    # Trimite ordinul către Crupier
    rezultat = crupier.executa_tranzactie(self, {
        'actiune': actiune,
        'companie': companie,
        'cantitate': cantitate
    })
    
    return rezultat
```

**Selecția companiilor:**
- Liber să aleagă orice companie din S&P 500
- Poate deține până la 20 de poziții simultan
- Nu există restricții de sector/diversificare (doar limitări de capital)

### 3.3 Costuri de Tranzacționare REALISTE (CRUCIAL)

```python
class CosturiRealiste:
    COMMISSION_PERCENT = 0.001        # 0.1% per tranzacție
    COMMISSION_FIXED = 1.0            # Minim $1 per tranzacție
    SPREAD_PERCENT = 0.0005           # 0.05% bid-ask spread
    SLIPPAGE_BASE = 0.0003            # 0.03% slippage de bază
    
    def cost_tranzactie(self, valoare, volum_ordine, volum_mediu_zilnic):
        # Slippage crește cu dimensiunea ordinului
        impact_piata = (volum_ordine / volum_mediu_zilnic) * 0.01
        slippage_total = self.SLIPPAGE_BASE + impact_piata
        
        cost_variabil = valoare * (
            self.COMMISSION_PERCENT + 
            self.SPREAD_PERCENT + 
            slippage_total
        )
        
        cost_total = max(cost_variabil, self.COMMISSION_FIXED)
        
        return cost_total

# Aplicat la FIECARE tranzacție (BUY și SELL)
```

**Constrângeri de lichiditate:**
```python
class ConstrangeriLichiditate:
    MAX_PERCENT_DAILY_VOLUME = 0.01  # Max 1% din volumul zilnic
    
    def limiteaza_ordine(self, cantitate_dorita, volum_mediu_zilnic):
        max_cantitate = volum_mediu_zilnic * self.MAX_PERCENT_DAILY_VOLUME
        
        if cantitate_dorita > max_cantitate:
            # Ordinul se execută parțial cu impact asupra prețului
            zile_necesare = ceil(cantitate_dorita / max_cantitate)
            pret_mediu_deteriorat = pret_curent * (1 + 0.002 * zile_necesare)
            return {
                'cantitate_executata': max_cantitate,
                'pret_executie': pret_mediu_deteriorat,
                'rest_ordine': cantitate_dorita - max_cantitate
            }
        else:
            return {'cantitate_executata': cantitate_dorita, 'pret_executie': pret_curent}
```

### 3.4 Recalibrare Trimestrială

```python
class RecalibrareDinamica:
    FRECVENTA = "trimestrial"  # La fiecare 3 luni
    MAX_SHIFT_PONDERE = 0.30   # Maximum 30% schimbare per indicator
    
    def permite_recalibrare(self, jucator, luna_curenta):
        if luna_curenta % 3 == 0:  # Luna 3, 6, 9, 12, etc.
            # Jucătorul poate recalcula ponderile
            # DAR DOAR pe datele PÂNĂ LA ACEL MOMENT
            date_disponibile = self.get_data_until(luna_curenta)
            ponderi_noi = jucator.recalculate_weights(date_disponibile)
            
            # Validare: schimbări prea drastice = red flag
            if self.validate_shift(ponderi_noi, jucator.ponderi_vechi):
                jucator.ponderi = ponderi_noi
                return True
            else:
                return False  # Respinge recalibrarea drastică
    
    def validate_shift(self, ponderi_noi, ponderi_vechi):
        for indicator in ponderi_noi.keys():
            shift = abs(ponderi_noi[indicator] - ponderi_vechi[indicator])
            if shift > self.MAX_SHIFT_PONDERE:
                return False
        return True
```

---

## PARTEA IV: METRICI DE EVALUARE (CELE MAI IMPORTANTE)

### 4.1 Metrici Risk-Adjusted (Scor Principal)

**NU se folosește profit brut! Se folosesc metrici risk-adjusted:**

```python
class MetriciAvansate:
    def sharpe_ratio(self, returns, risk_free_rate=0.02):
        """
        Returnuri ajustate la risc (standard în industrie)
        Valori bune: > 1.0, Excelent: > 2.0
        """
        excess_returns = returns - (risk_free_rate / 252)  # Daily risk-free rate
        if len(returns) == 0 or np.std(returns) == 0:
            return 0
        return np.mean(excess_returns) / np.std(returns) * np.sqrt(252)
    
    def sortino_ratio(self, returns, risk_free_rate=0.02):
        """
        Similar cu Sharpe dar penalizează doar volatilitatea descendentă
        Mai relevant pentru investitori (downside risk matters more)
        """
        excess_returns = returns - (risk_free_rate / 252)
        downside_returns = returns[returns < 0]
        if len(downside_returns) == 0:
            return float('inf')
        downside_std = np.std(downside_returns)
        if downside_std == 0:
            return 0
        return np.mean(excess_returns) / downside_std * np.sqrt(252)
    
    def max_drawdown(self, equity_curve):
        """
        Cea mai mare scădere de la peak la trough
        Măsoară cât de mult poate pierde strategia în worst case
        Valori acceptabile: < -25%
        """
        running_max = np.maximum.accumulate(equity_curve)
        drawdown = (equity_curve - running_max) / running_max
        return np.min(drawdown)
    
    def calmar_ratio(self, returns, equity_curve):
        """
        Return anual / Max Drawdown
        Măsoară reward-to-risk
        Valori bune: > 0.5
        """
        annual_return = np.mean(returns) * 252
        max_dd = abs(self.max_drawdown(equity_curve))
        if max_dd == 0:
            return float('inf')
        return annual_return / max_dd
    
    def win_rate(self, trades):
        """
        Procentul de tranzacții profitabile
        Bonus pentru consistență
        """
        profitable = sum(1 for t in trades if t['profit'] > 0)
        return profitable / len(trades) if len(trades) > 0 else 0
    
    def profit_factor(self, trades):
        """
        Total profit / Total loss
        Valori bune: > 1.5
        """
        gross_profit = sum(t['profit'] for t in trades if t['profit'] > 0)
        gross_loss = abs(sum(t['profit'] for t in trades if t['profit'] < 0))
        if gross_loss == 0:
            return float('inf')
        return gross_profit / gross_loss
    
    def scor_compozit(self, returns, equity_curve, trades):
        """
        SCORUL FINAL - combinație ponderată
        """
        sharpe = self.sharpe_ratio(returns)
        sortino = self.sortino_ratio(returns)
        calmar = self.calmar_ratio(returns, equity_curve)
        win_rate = self.win_rate(trades)
        profit_factor = self.profit_factor(trades)
        
        # Normalizare și combinare
        scor = (
            0.25 * min(sharpe / 2.0, 1.0) +           # Sharpe normalizat
            0.25 * min(sortino / 2.5, 1.0) +          # Sortino normalizat
            0.20 * min(calmar / 1.0, 1.0) +           # Calmar normalizat
            0.15 * win_rate +                          # Win rate direct
            0.15 * min(profit_factor / 2.0, 1.0)      # Profit factor normalizat
        )
        
        return scor  # Valoare între 0 și 1
```

### 4.2 Penalizare pentru Complexitate

```python
class RegularizareComplexitate:
    def penalizare(self, numar_indicatori):
        """
        Principiul Occam's Razor: strategii simple sunt preferabile
        Penalizare crescătoare cu numărul de indicatori
        """
        if numar_indicatori <= 5:
            penalty = 0.0
        elif numar_indicatori <= 10:
            penalty = 0.05
        elif numar_indicatori <= 15:
            penalty = 0.10
        else:
            penalty = 0.15
        
        return penalty
    
    def scor_ajustat_complexitate(self, scor_compozit, numar_indicatori):
        penalty = self.penalizare(numar_indicatori)
        return scor_compozit * (1 - penalty)
```

### 4.3 Evaluare pe Regimuri de Piață

```python
class RegimeEvaluation:
    def identifica_regimuri(self, sp500_returns, vix_data):
        """
        Clasifică fiecare perioadă în regimuri de piață
        """
        regimuri = {
            'bull_market': [],       # Trend > +15% anual
            'bear_market': [],       # Trend < -10% anual
            'high_volatility': [],   # VIX > 25
            'low_volatility': [],    # VIX < 15
            'sideways': []           # -10% < Trend < +10%
        }
        
        # Pentru fiecare lună, clasifică regimul
        for luna in range(len(sp500_returns)):
            trend_3m = calculate_trend(sp500_returns, luna, window=3)
            vix_avg = vix_data[luna]
            
            if vix_avg > 25:
                regimuri['high_volatility'].append(luna)
            elif vix_avg < 15:
                regimuri['low_volatility'].append(luna)
            
            if trend_3m > 0.15:
                regimuri['bull_market'].append(luna)
            elif trend_3m < -0.10:
                regimuri['bear_market'].append(luna)
            else:
                regimuri['sideways'].append(luna)
        
        return regimuri
    
    def scor_multiregim(self, jucator, regimuri):
        """
        Strategia trebuie să funcționeze în TOATE regimurile
        """
        scoruri_pe_regim = {}
        
        for regim, luni in regimuri.items():
            returns_regim = jucator.get_returns_for_months(luni)
            scoruri_pe_regim[regim] = MetriciAvansate().scor_compozit(returns_regim, ...)
        
        # Scorul final = media dar PENALIZAT de cel mai slab regim
        scor_mediu = np.mean(list(scoruri_pe_regim.values()))
        scor_minim = min(scoruri_pe_regim.values())
        
        # 60% medie, 40% worst case
        return 0.6 * scor_mediu + 0.4 * scor_minim
```

### 4.4 SCORUL FINAL LUNAR (pentru eliminări)

```python
def scor_final_lunar(jucator, luna):
    """
    Scorul folosit pentru clasament și eliminări
    """
    returns = jucator.get_returns_until(luna)
    equity_curve = jucator.get_equity_curve_until(luna)
    trades = jucator.get_trades_until(luna)
    
    # 1. Scor compozit risk-adjusted
    scor_compozit = MetriciAvansate().scor_compozit(returns, equity_curve, trades)
    
    # 2. Ajustare pentru complexitate
    scor_ajustat = RegularizareComplexitate().scor_ajustat_complexitate(
        scor_compozit, 
        len(jucator.indicatori)
    )
    
    # 3. Evaluare multi-regim (dacă au trecut suficiente luni)
    if luna >= 6:
        regimuri = RegimeEvaluation().identifica_regimuri(...)
        scor_regimuri = RegimeEvaluation().scor_multiregim(jucator, regimuri)
        scor_final = 0.7 * scor_ajustat + 0.3 * scor_regimuri
    else:
        scor_final = scor_ajustat
    
    return scor_final
```

---

## PARTEA V: REGULI DE ELIMINARE

### 5.1 Regula de Eliminare Lunară

```python
class EliminareJucatori:
    PRAG_ELIMINARE = 0.20  # 20% distanță față de penultim
    
    def evalueaza_eliminare(self, jucatori_activi, luna):
        """
        La sfârșitul fiecărei luni
        """
        if len(jucatori_activi) <= 2:
            return None  # Nu elimina dacă rămân doar 2
        
        # Calculează scoruri finale
        scoruri = [(j, scor_final_lunar(j, luna)) for j in jucatori_activi]
        scoruri_sorted = sorted(scoruri, key=lambda x: x[1], reverse=True)
        
        ultim = scoruri_sorted[-1]
        penultim = scoruri_sorted[-2]
        
        # Calculează distanța relativă
        distanta = (penultim[1] - ultim[1]) / penultim[1]
        
        if distanta >= self.PRAG_ELIMINARE:
            return ultim[0]  # Elimină ultimul jucător
        else:
            return None  # Nimeni eliminat
```

### 5.2 Condiții Suplimentare de Eliminare

```python
class EliminariSuplimentare:
    def verifica_incalcari(self, jucator, luna):
        """
        Eliminare imediată pentru încălcări grave
        """
        # 1. Nu a făcut tranzacții obligatorii
        if jucator.get_num_trades_in_month(luna) == 0:
            return "ELIMINAT: Nu a efectuat tranzacția lunară obligatorie"
        
        # 2. Drawdown catastrofal (pierdere > 50%)
        equity = jucator.get_equity_at(luna)
        if equity < 50000:  # Mai puțin de 50% din capital inițial
            return "ELIMINAT: Drawdown > 50%"
        
        # 3. Overtrading excesiv
        if jucator.get_num_trades_in_month(luna) > 50:
            return "ELIMINAT: Overtrading (> 50 tranzacții/lună)"
        
        return None  # OK
```

---

## PARTEA VI: VALIDARE STATISTICĂ (TEST SET)

### 6.1 Evaluare Out-of-Sample

```python
class ValidareStatistica:
    def evalueaza_test_set(self, jucatori_supravietuitori):
        """
        După încheierea jocului pe VALIDATION SET,
        evaluăm pe TEST SET (date necunoscute)
        """
        rezultate_finale = []
        
        for jucator in jucatori_supravietuitori:
            # Rulează strategia pe TEST SET (2022-2023)
            # Jucătorul folosește EXACT aceleași ponderi finale
            # NU are voie să recalibreze!
            
            returns_test = self.simuleaza_strategie(jucator, self.date_test)
            equity_curve_test = self.get_equity_curve(returns_test)
            trades_test = self.get_trades(jucator, self.date_test)
            
            # Calculează scor compozit pe TEST
            scor_test = MetriciAvansate().scor_compozit(
                returns_test, 
                equity_curve_test, 
                trades_test
            )
            
            # VALIDARE STATISTICĂ
            validare = self.bootstrap_validation(jucator, returns_test)
            
            rezultate_finale.append({
                'jucator': jucator,
                'scor_test': scor_test,
                'validare_statistica': validare
            })
        
        return rezultate_finale
    
    def bootstrap_validation(self, jucator, returns_test):
        """
        Verifică dacă performanța este semnificativă statistic
        """
        n_iterations = 1000
        bootstrapped_sharpe = []
        
        for _ in range(n_iterations):
            # Resample cu replacement
            sample_returns = np.random.choice(
                returns_test, 
                size=len(returns_test), 
                replace=True
            )
            sharpe_sample = MetriciAvansate().sharpe_ratio(sample_returns)
            bootstrapped_sharpe.append(sharpe_sample)
        
        # Confidence Interval 95%
        ci_lower = np.percentile(bootstrapped_sharpe, 2.5)
        ci_upper = np.percentile(bootstrapped_sharpe, 97.5)
        
        # Performanța este semnificativă dacă CI nu include 0
        is_significant = ci_lower > 0
        
        return {
            'sharpe_observed': MetriciAvansate().sharpe_ratio(returns_test),
            'confidence_interval': (ci_lower, ci_upper),
            'is_significant': is_significant,
            'p_value': self.calculate_p_value(bootstrapped_sharpe, 0)
        }
    
    def permutation_test_vs_market(self, returns_strategy, returns_sp500):
        """
        Testează dacă strategia bate piața SEMNIFICATIV
        """
        observed_diff = np.mean(returns_strategy) - np.mean(returns_sp500)
        
        permutation_diffs = []
        combined = np.concatenate([returns_strategy, returns_sp500])
        
        for _ in range(1000):
            np.random.shuffle(combined)
            perm_strategy = combined[:len(returns_strategy)]
            perm_market = combined[len(returns_strategy):]
            diff = np.mean(perm_strategy) - np.mean(perm_market)
            permutation_diffs.append(diff)
        
        p_value = np.mean(np.abs(permutation_diffs) >= np.abs(observed_diff))
        
        return {
            'beats_market': observed_diff > 0,
            'is_significant': p_value < 0.05,
            'p_value': p_value
        }
```

### 6.2 Stress Testing

```python
class StressTesting:
    def define_stress_scenarios(self):
        """
        Scenarii extreme istorice
        """
        return {
            'financial_crisis_2008': {
                'period': '2008-09 to 2009-03',
                'sp500_return': -0.40,
                'duration_months': 6
            },
            'covid_crash_2020': {
                'period': '2020-02 to 2020-03',
                'sp500_return': -0.34,
                'duration_months': 1
            },
            'dot_com_burst_2000': {
                'period': '2000-03 to 2002-10',
                'sp500_return': -0.49,
                'duration_months': 31
            }
        }
    
    def evalueaza_rezilienta(self, jucator, scenarii):
        """
        Testează cum supraviețuiește strategia în crize
        """
        rezultate_stress = {}
        
        for nume_scenariu, parametri in scenarii.items():
            # Simulează strategia în acel scenariu
            max_dd = self.simuleaza_scenariu(jucator, parametri)
            
            # Criteriu de supraviețuire: Max DD < -50%
            survives = max_dd > -0.50
            
            rezultate_stress[nume_scenariu] = {
                'max_drawdown': max_dd,
                'survives': survives
            }
        
        # Strategia trebuie să supraviețuiască TOATE scenariile
        supravietuieste_toate = all(
            r['survives'] for r in rezultate_stress.values()
        )
        
        return {
            'rezultate_detaliate': rezultate_stress,
            'pass_stress_test': supravietuieste_toate
        }
```

### 6.3 Determinarea Câștigătorului FINAL

```python
def determina_castigator_final(jucatori_supravietuitori, date_test):
    """
    Câștigătorul REAL nu este cel din VALIDATION
    ci cel care performează cel mai bine pe TEST + validare statistică
    """
    validare = ValidareStatistica()
    stress = StressTesting()
    
    scoruri_finale = []
    
    for jucator in jucatori_supravietuitori:
        # 1. Performanță pe TEST SET
        rezultate_test = validare.evalueaza_test_set([jucator])[0]
        scor_test = rezultate_test['scor_test']
        
        # 2. Validare statistică
        validare_stat = rezultate_test['validare_statistica']
        is_significant = validare_stat['is_significant']
        
        # 3. Permutation test vs Market
        returns_test = get_returns(jucator, date_test)
        returns_sp500 = get_sp500_returns(date_test)
        market_test = validare.permutation_test_vs_market(returns_test, returns_sp500)
        beats_market = market_test['is_significant']
        
        # 4. Stress testing
        scenarii = stress.define_stress_scenarios()
        stress_results = stress.evalueaza_rezilienta(jucator, scenarii)
        pass_stress = stress_results['pass_stress_test']
        
        # 5. SCOR FINAL COMPOZIT
        scor_final = (
            0.40 * scor_test +                      # Performanță pe test set (40%)
            0.25 * float(is_significant) +          # Semnificație statistică (25%)
            0.20 * float(beats_market) +            # Bate piața semnificativ (20%)
            0.15 * float(pass_stress)               # Supraviețuire stress test (15%)
        )
        
        scoruri_finale.append({
            'jucator': jucator,
            'scor_final': scor_final,
            'breakdown': {
                'scor_test': scor_test,
                'is_statistically_significant': is_significant,
                'beats_market': beats_market,
                'passes_stress_test': pass_stress
            }
        })
    
    # Sortează și determină câștigătorul
    scoruri_finale.sort(key=lambda x: x['scor_final'], reverse=True)
    
    return scoruri_finale[0]  # Câștigătorul
```

---

## PARTEA VII: OUTPUT ȘI RAPORTARE

### 7.1 Raport Complet pentru Câștigător

```python
class RaportFinal:
    def genereaza_raport_castigator(self, castigator):
        """
        Generează raport detaliat despre strategia câștigătoare
        """
        return {
            'identitate': {
                'jucator_id': castigator.id,
                'metoda_declarata': castigator.metoda_descriere
            },
            
            'ponderi_finale': {
                indicator: pondere 
                for indicator, pondere in castigator.ponderi.items()
            },
            
            'performanta_validation': {
                'sharpe_ratio': calculate_sharpe(castigator, 'validation'),
                'sortino_ratio': calculate_sortino(castigator, 'validation'),
                'max_drawdown': calculate_max_dd(castigator, 'validation'),
                'total_return': calculate_total_return(castigator, 'validation'),
                'win_rate': calculate_win_rate(castigator, 'validation')
            },
            
            'performanta_test': {
                'sharpe_ratio': calculate_sharpe(castigator, 'test'),
                'sortino_ratio': calculate_sortino(castigator, 'test'),
                'max_drawdown': calculate_max_dd(castigator, 'test'),
                'total_return': calculate_total_return(castigator, 'test'),
                'win_rate': calculate_win_rate(castigator, 'test'),
                'confidence_interval': castigator.validare_statistica['confidence_interval'],
                'beats_market': castigator.validare_statistica['beats_market']
            },
            
            'breakdown_pe_regimuri': {
                regim: calculate_performance(castigator, regim)
                for regim in ['bull_market', 'bear_market', 'high_vol', 'low_vol']
            },
            
            'stress_test_results': castigator.stress_test_results,
            
            'indicatori_importanti': self.analiza_importanta_indicatori(castigator),
            
            'date_externe_folosite': [
                {
                    'sursa': sursa,
                    'tip': tip,
                    'impact_estimat': self.estimeaza_impact(castigator, sursa)
                }
                for sursa, tip in castigator.surse_externe.items()
            ],
            
            'tranzactii_notabile': self.identifica_tranzactii_cheie(castigator),
            
            'lectii_invatate': self.extrage_insights(castigator),
            
            'recomandari_implementare': self.ghid_implementare_practica(castigator)
        }
    
    def analiza_importanta_indicatori(self, jucator):
        """
        Determină care indicatori au avut cel mai mare impact
        folosind ablation study
        """
        importance = {}
        baseline = jucator.performanta_totala
        
        for indicator in jucator.indicatori:
            # Simulează performanța FĂRĂ acest indicator
            perf_fara = self.simuleaza_fara_indicator(jucator, indicator)
            impact = baseline - perf_fara
            importance[indicator] = {
                'impact_absolut': impact,
                'pondere': jucator.ponderi[indicator],
                'eficienta': impact / jucator.ponderi[indicator]  # Impact per unitate de pondere
            }
        
        # Sortează după impact
        return dict(sorted(importance.items(), key=lambda x: x[1]['impact_absolut'], reverse=True))
```

### 7.2 Analiza Comparativă a Tuturor Jucătorilor

```python
class AnalizaComparativa:
    def compara_toti_jucatorii(self, toti_jucatorii):
        """
        Compară toți jucătorii (incluși cei eliminați) pentru învățare
        """
        return {
            'clasament_final': self.genereaza_clasament(toti_jucatorii),
            
            'pattern-uri_comune_la_castigatori': self.identifica_pattern_castigatori(
                toti_jucatorii[:3]  # Top 3
            ),
            
            'greseli_frecvente_la_eliminati': self.identifica_greseli(
                [j for j in toti_jucatorii if j.eliminat]
            ),
            
            'metode_performante': self.ranking_metode(toti_jucatorii),
            
            'indicatori_populari': self.statistici_indicatori(toti_jucatorii),
            
            'surse_externe_eficiente': self.analiza_surse_externe(toti_jucatorii),
            
            'corelatie_complexitate_performanta': self.analiza_complexitate(toti_jucatorii),
            
            'evolutie_in_timp': self.grafic_evolutie_toti(toti_jucatorii)
        }
    
    def identifica_pattern_castigatori(self, top_jucatori):
        """
        Ce au în comun strategiile de succes?
        """
        patterns = {
            'indicatori_comuni': set.intersection(*[
                set(j.indicatori) for j in top_jucatori
            ]),
            
            'range_ponderi_comune': {
                indicator: {
                    'min': min(j.ponderi.get(indicator, 0) for j in top_jucatori),
                    'max': max(j.ponderi.get(indicator, 0) for j in top_jucatori),
                    'avg': np.mean([j.ponderi.get(indicator, 0) for j in top_jucatori])
                }
                for indicator in set().union(*[set(j.indicatori) for j in top_jucatori])
            },
            
            'frecventa_recalibrari': np.mean([
                j.numar_recalibrari for j in top_jucatori
            ]),
            
            'utilizare_date_externe': {
                'procent_cu_date_externe': sum(
                    1 for j in top_jucatori if j.surse_externe
                ) / len(top_jucatori),
                'surse_comune': set.intersection(*[
                    set(j.surse_externe.keys()) for j in top_jucatori if j.surse_externe
                ]) if any(j.surse_externe for j in top_jucatori) else set()
            },
            
            'profile_risc': {
                'avg_sharpe': np.mean([j.sharpe_ratio for j in top_jucatori]),
                'avg_max_dd': np.mean([j.max_drawdown for j in top_jucatori]),
                'avg_win_rate': np.mean([j.win_rate for j in top_jucatori])
            }
        }
        
        return patterns
```

### 7.3 Recomandări pentru Viitor

```python
class RecomandariViitor:
    def genereaza_recomandari(self, analiza_completa):
        """
        Bazat pe rezultatele jocului, ce am învățat pentru viitor?
        """
        return {
            'ponderi_recomandate_baseline': self.extrage_ponderi_optime(
                analiza_completa['pattern-uri_comune_la_castigatori']
            ),
            
            'indicatori_essentiali': self.lista_indicatori_critici(
                analiza_completa
            ),
            
            'indicatori_redundanti': self.lista_indicatori_inutili(
                analiza_completa
            ),
            
            'best_practices': [
                'Folosește între 5-10 indicatori (sweet spot)',
                'Recalibrează trimestrial, dar nu dramatic (max 30% shift)',
                'Combină date tehnice + fundamentale pentru best results',
                f"Indicatori must-have: {analiza_completa['indicatori_must_have']}",
                f"Evită overtrading: keep under {analiza_completa['avg_trades_winners']} trades/month",
                'Stress test obligatoriu înainte de deployment'
            ],
            
            'surse_date_utile': self.ranking_surse_externe(analiza_completa),
            
            'metode_recomandate': self.top_metode_per_categorie(analiza_completa),
            
            'red_flags': [
                f"Drawdown > {analiza_completa['max_acceptable_dd']}",
                f"Complexitate > {analiza_completa['max_indicators']} indicatori",
                'Sharpe ratio < 1.0 pe validation',
                'Performanță inconsistentă între regimuri',
                'Fail stress test'
            ]
        }
```

---

## PARTEA VIII: IMPLEMENTARE PRACTICĂ

### 8.1 Arhitectura Sistemului

```python
class TradingGameSystem:
    def __init__(self):
        self.crupier = Crupier()
        self.jucatori = []
        self.date = self.incarca_date_segmentate()
        self.validare_stats = ValidareStatistica()
        self.stress_test = StressTesting()
    
    def run_complete_game(self):
        """
        Execută întregul joc de la început până la sfârșit
        """
        print("=" * 80)
        print("PORNIRE JOC DE TRADING COMPETITIV")
        print("=" * 80)
        
        # FAZA 1: Înregistrare și Pregătire
        print("\n[FAZA 1] Înregistrare jucători și calculare ponderi...")
        self.faza_pregatire()
        
        # FAZA 2: Competiție pe VALIDATION SET
        print("\n[FAZA 2] Competiție pe Validation Set (24 luni)...")
        supravietuitori = self.faza_competitie()
        
        # FAZA 3: Evaluare pe TEST SET
        print("\n[FAZA 3] Evaluare finală pe Test Set...")
        castigator = self.faza_evaluare_finala(supravietuitori)
        
        # FAZA 4: Raportare și Analiză
        print("\n[FAZA 4] Generare rapoarte și învățare...")
        self.faza_raportare(castigator)
        
        return castigator
    
    def faza_pregatire(self):
        """Înregistrare, solicitare indicatori, calculare ponderi"""
        for jucator in self.jucatori:
            # Declarare metodă (opțional)
            if jucator.doreste_sa_declare_metoda():
                metoda = jucator.descrie_metoda()
                self.crupier.inregistreaza_metoda(jucator, metoda)
            
            # Solicitare indicatori
            indicatori_doriti = jucator.selecteaza_indicatori()
            date_indicatori = self.crupier.furnizeaza_indicatori(
                jucator, 
                indicatori_doriti, 
                self.date['training']
            )
            
            # Solicitare surse externe (opțional)
            if jucator.doreste_date_externe():
                surse = jucator.solicita_surse_externe()
                for sursa in surse:
                    if self.crupier.aproba_sursa(sursa):
                        date_externe = jucator.procura_date_externe(sursa)
                        self.crupier.inregistreaza_date_externe(jucator, sursa, date_externe)
            
            # Calculare ponderi pe training set
            jucator.calculeaza_ponderi_initiale(date_indicatori)
            
            print(f"  ✓ Jucător {jucator.id} pregătit: {len(jucator.indicatori)} indicatori")
    
    def faza_competitie(self):
        """24 luni de competiție cu eliminări"""
        jucatori_activi = self.jucatori.copy()
        
        for luna in range(1, 25):  # 24 luni
            print(f"\n--- Luna {luna}/24 ---")
            
            # Recalibrare trimestrială
            if luna % 3 == 0:
                print(f"  Punct de recalibrare trimestrială...")
                for jucator in jucatori_activi:
                    if jucator.doreste_recalibrare():
                        ponderi_noi = jucator.recalculeaza_ponderi(
                            self.date['validation'][:luna]
                        )
                        if self.crupier.valideaza_recalibrare(jucator.ponderi, ponderi_noi):
                            jucator.ponderi = ponderi_noi
            
            # Fiecare jucător ia decizii și execută tranzacții
            for jucator in jucatori_activi:
                decizii = jucator.decide_tranzactii_luna(luna, self.date['validation'])
                
                for decizie in decizii:
                    rezultat = self.crupier.executa_tranzactie(jucator, decizie)
                    jucator.proceseaza_rezultat(rezultat)
            
            # Evaluare și eliminare
            for jucator in jucatori_activi:
                incalcare = self.crupier.verifica_incalcari(jucator, luna)
                if incalcare:
                    print(f"  ✗ {jucator.id} ELIMINAT: {incalcare}")
                    jucatori_activi.remove(jucator)
            
            # Eliminare bazată pe performanță
            if len(jucatori_activi) > 2:
                eliminat = self.crupier.evalueaza_eliminare(jucatori_activi, luna)
                if eliminat:
                    print(f"  ✗ {eliminat.id} ELIMINAT: sub prag 20%")
                    jucatori_activi.remove(eliminat)
            
            # Afișare clasament
            self.afiseaza_clasament(jucatori_activi, luna)
        
        print(f"\n{'='*80}")
        print(f"SUPRAVIEȚUITORI: {[j.id for j in jucatori_activi]}")
        print(f"{'='*80}")
        
        return jucatori_activi
    
    def faza_evaluare_finala(self, supravietuitori):
        """Evaluare pe TEST SET + validare statistică"""
        rezultate = []
        
        for jucator in supravietuitori:
            print(f"\n  Evaluare {jucator.id} pe Test Set...")
            
            # Rulează pe test set (ponderi fixe, NU recalibrare)
            returns_test = self.simuleaza_pe_test(jucator, self.date['test'])
            
            # Validare statistică
            bootstrap = self.validare_stats.bootstrap_validation(jucator, returns_test)
            market_test = self.validare_stats.permutation_test_vs_market(
                returns_test,
                self.date['test_sp500_returns']
            )
            stress_rezultate = self.stress_test.evalueaza_rezilienta(jucator, scenarii)
            
            scor_final = self.calculeaza_scor_final(
                jucator, returns_test, bootstrap, market_test, stress_rezultate
            )
            
            rezultate.append({
                'jucator': jucator,
                'scor': scor_final,
                'detalii': {
                    'bootstrap': bootstrap,
                    'market_test': market_test,
                    'stress_test': stress_rezultate
                }
            })
        
        # Sortează și declară câștigătorul
        rezultate.sort(key=lambda x: x['scor'], reverse=True)
        castigator = rezultate[0]['jucator']
        
        print(f"\n{'='*80}")
        print(f"🏆 CÂȘTIGĂTOR: {castigator.id}")
        print(f"{'='*80}")
        
        return castigator
    
    def faza_raportare(self, castigator):
        """Generare rapoarte comprehensive"""
        raport = RaportFinal()
        analiza = AnalizaComparativa()
        recomandari = RecomandariViitor()
        
        # Raport câștigător
        raport_castigator = raport.genereaza_raport_castigator(castigator)
        self.salveaza_raport(raport_castigator, 'castigator_report.json')
        
        # Analiză comparativă
        analiza_completa = analiza.compara_toti_jucatorii(self.jucatori)
        self.salveaza_raport(analiza_completa, 'comparative_analysis.json')
        
        # Recomandări pentru viitor
        recomandari_finale = recomandari.genereaza_recomandari(analiza_completa)
        self.salveaza_raport(recomandari_finale, 'recommendations.json')
        
        print("\n✓ Rapoarte generate cu succes!")
        print(f"  - castigator_report.json")
        print(f"  - comparative_analysis.json")
        print(f"  - recommendations.json")
```

### 8.2 Exemple de Implementare pentru Jucători

```python
# Exemplu 1: Jucător cu Random Forest + News Sentiment
class JucatorRandomForestNews:
    def __init__(self, id):
        self.id = id
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.indicatori = ['RSI', 'MACD', 'Bollinger_Bands', 'ATR', 'Volume_SMA']
        self.surse_externe = {}
        self.model = None
        self.ponderi = {}
    
    def descrie_metoda(self):
        return "Random Forest pentru predicție + News Sentiment Analysis"
    
    def solicita_surse_externe(self):
        return [{
            'tip_sursa': 'news',
            'descriere': 'NewsAPI pentru sentiment analysis',
            'frecventa': 'zilnic'
        }]
    
    def calculeaza_ponderi_initiale(self, date_indicatori):
        # Pregătește features pentru RF
        X = self.prepare_features(date_indicatori)
        y = self.prepare_labels(date_indicatori)  # Future returns
        
        # Train Random Forest
        from sklearn.ensemble import RandomForestRegressor
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.model.fit(X, y)
        
        # Extract feature importances ca ponderi
        importances = self.model.feature_importances_
        total = sum(importances)
        self.ponderi = {
            self.indicatori[i]: importances[i] / total
            for i in range(len(self.indicatori))
        }
    
    def decide_tranzactii_luna(self, luna, date_validation):
        decizii = []
        
        # Pentru fiecare companie din watchlist
        for companie in self.select_watchlist():
            # Get technical indicators
            tech_signal = self.calculate_technical_signal(companie, luna, date_validation)
            
            # Get news sentiment
            news_data = self.get_news_data(companie, luna)
            sentiment_signal = self.analyze_sentiment(news_data)
            
            # Combine: 80% technical, 20% sentiment
            final_signal = 0.8 * tech_signal + 0.2 * sentiment_signal
            
            if final_signal > 0.5:
                decizii.append({
                    'actiune': 'BUY',
                    'companie': companie,
                    'cantitate': self.size_position(final_signal)
                })
            elif final_signal < -0.5:
                decizii.append({
                    'actiune': 'SELL',
                    'companie': companie,
                    'cantitate': self.size_position(abs(final_signal))
                })
        
        return decizii


# Exemplu 2: Jucător cu Deep Learning pe GPU
class JucatorDeepLearningGPU:
    def __init__(self, id):
        self.id = id
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.indicatori = ['SMA_20', 'SMA_50', 'RSI', 'MACD', 'ATR', 'Volume']
        self.lstm_model = None
        self.ponderi = {}
    
    def descrie_metoda(self):
        return "LSTM Neural Network pe GPU pentru predicție time-series"
    
    def calculeaza_ponderi_initiale(self, date_indicatori):
        # Creează și antrenează LSTM pe GPU
        self.lstm_model = self.build_lstm().to(self.device)
        
        # Pregătește sequences pentru LSTM
        X_seq, y_seq = self.prepare_sequences(date_indicatori)
        X_seq = torch.tensor(X_seq).float().to(self.device)
        y_seq = torch.tensor(y_seq).float().to(self.device)
        
        # Training pe GPU
        optimizer = torch.optim.Adam(self.lstm_model.parameters(), lr=0.001)
        criterion = torch.nn.MSELoss()
        
        for epoch in range(100):
            optimizer.zero_grad()
            outputs = self.lstm_model(X_seq)
            loss = criterion(outputs, y_seq)
            loss.backward()
            optimizer.step()
        
        # Extract attention weights ca ponderi
        attention_weights = self.extract_attention_weights()
        total = sum(attention_weights.values())
        self.ponderi = {
            k: v/total for k, v in attention_weights.items()
        }
    
    def build_lstm(self):
        class LSTMAttention(torch.nn.Module):
            def __init__(self, input_size, hidden_size, num_layers):
                super().__init__()
                self.lstm = torch.nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
                self.attention = torch.nn.Linear(hidden_size, 1)
                self.fc = torch.nn.Linear(hidden_size, 1)
            
            def forward(self, x):
                lstm_out, _ = self.lstm(x)
                attn_weights = torch.softmax(self.attention(lstm_out), dim=1)
                context = torch.sum(attn_weights * lstm_out, dim=1)
                output = self.fc(context)
                return output
        
        return LSTMAttention(input_size=len(self.indicatori), hidden_size=64, num_layers=2)


# Exemplu 3: Jucător cu Genetic Algorithm + Fundamentals
class JucatorGeneticFundamentals:
    def __init__(self, id):
        self.id = id
        self.indicatori = ['RSI', 'MACD', 'EMA_12', 'EMA_26', 'Bollinger_Bands']
        self.ponderi = {}
        self.surse_externe = {}
    
    def descrie_metoda(self):
        return "Genetic Algorithm pentru optimizare + SEC Filings pentru fundamentals"
    
    def solicita_surse_externe(self):
        return [{
            'tip_sursa': 'financial_reports',
            'descriere': 'SEC EDGAR pentru 10-K și 10-Q',
            'frecventa': 'trimestrial'
        }]
    
    def calculeaza_ponderi_initiale(self, date_indicatori):
        # Genetic Algorithm pentru găsirea ponderilor optime
        from deap import base, creator, tools, algorithms
        
        # Setup GA
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)
        
        toolbox = base.Toolbox()
        toolbox.register("attr_float", random.random)
        toolbox.register("individual", tools.initRepeat, creator.Individual,
                        toolbox.attr_float, n=len(self.indicatori))
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        
        def evaluate_weights(individual):
            # Normalizează ponderile
            total = sum(individual)
            ponderi_norm = {
                self.indicatori[i]: individual[i]/total
                for i in range(len(individual))
            }
            
            # Backtest cu aceste ponderi
            sharpe = self.backtest_with_weights(ponderi_norm, date_indicatori)
            return (sharpe,)
        
        toolbox.register("evaluate", evaluate_weights)
        toolbox.register("mate", tools.cxBlend, alpha=0.5)
        toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.2, indpb=0.2)
        toolbox.register("select", tools.selTournament, tournsize=3)
        
        # Rulează GA
        pop = toolbox.population(n=50)
        algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=100, verbose=False)
        
        # Extrage cel mai bun individ
        best_individual = tools.selBest(pop, k=1)[0]
        total = sum(best_individual)
        self.ponderi = {
            self.indicatori[i]: best_individual[i]/total
            for i in range(len(self.indicatori))
        }
```

---

## REZUMAT FINAL

### Flux Complet al Jocului

```
1. PREGĂTIRE (Training Set: 2015-2019)
   ├─ Jucătorii se înregistrează
   ├─ Declară metode (opțional)
   ├─ Solicită indicatori tehnici
   ├─ Solicită surse externe (opțional)
   ├─ Calculează ponderi inițiale
   └─ Raportează totul către Crupier

2. COMPETIȚIE (Validation Set: 2020-2021, 24 luni)
   ├─ Pentru fiecare lună:
   │  ├─ Recalibrare trimestrială (opțional, max 30% shift)
   │  ├─ Jucătorii decid tranzacții bazat pe ponderi
   │  ├─ Trimite ordine → Crupier
   │  ├─ Crupier execută cu costuri realiste
   │  ├─ Crupier verifică încălcări
   │  └─ Crupier elimină ultimul (dacă >20% distanță)
   └─ Supraviețuitori = cei rămas după 24 luni

3. EVALUARE FINALĂ (Test Set: 2022-2023)
   ├─ Rulare pe date NECUNOSCUTE
   ├─ Ponderi FIXE (no recalibrare)
   ├─ Bootstrap validation (CI 95%)
   ├─ Permutation test vs S&P500
   ├─ Stress testing (scenarii extreme)
   └─ Scor final compozit → CÂȘTIGĂTOR

4. RAPORTARE
   ├─ Raport detaliat câștigător
   ├─ Analiză comparativă toți jucătorii
   ├─ Pattern-uri de succes
   ├─ Recomandări pentru viitor
   └─ Ponderi optime descoperite
```

### Caracteristici Cheie

✅ **Realism maxim**: Costuri tranzacționare, lichiditate, slippage  
✅ **Validare riguroasă**: Train/Val/Test split, bootstrap, permutation tests  
✅ **Flexibilitate**: Libertate totală în metode, combinații, date externe  
✅ **Accelerare GPU**: Suport pentru deep learning și optimizări masive  
✅ **Transparență**: Toate tranzacțiile și datele auditate de Crupier  
✅ **Metrici avansate**: Sharpe, Sortino, Calmar, Max DD, nu profit brut  
✅ **Robustețe**: Testare pe multiple regimuri + stress testing  
✅ **Învățare**: Meta-analiză pentru descoperirea pattern-urilor de succes  

### Scopul Final

**Descoperirea ponderilor optime pentru indicatori tehnici care:**
- Funcționează în REALITATE (nu doar pe date istorice)
- Sunt robuste în multiple regimuri de piață
- Supraviețuiesc scenarii extreme
- Bat piața semnificativ statistic
- Sunt simple și interpretabile
- Pot fi implementate practic

---

**FIN PROMPT**