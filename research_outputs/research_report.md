# Reproduzierbarer Research-/Backtest-Report

## Laufparameter
- as_of: `2026-09-17`
- warmup_bdays: `300`
- min_trades: `40`
- baseline_config: `config_best_2011_2026.json`
- baseline_config_hash: `033c31ccd2f184088ff286a6ce701f17f53fb8f02bed8f524605038311ebf0a3`

## Baseline (produktionsnah, unverändert im Live-Scanner)
- 55-Tage-Breakout
- Regimefilter Close > SMA200
- Momentum 126 Tage
- ATR-Stop / ATR-Trailing laut Baseline-Config
- Entry-Annahme: Signal am Close, Entry frühestens nächster handelbarer Open
- Kosten: spread_bps_per_side je Seite

## Survivorship-Bias und Datenlimitierung
- Belastbare Tests benötigen **PIT-Mitgliedschaften** und **historische delistingbereinigte Preisreihen**.
- Läufe mit aktuellen Wikipedia-Komponenten + Yahoo-Daten sind `survivorship_biased=true` und **kein belastbarer Performance-Nachweis**.

## Laufnotizen
- sp500: 1 Symbole ohne Preisdaten (Beispiele: SPY)
- sp500: Benchmark/Universum konnte nicht geladen werden; Ergebnisse ausgelassen.
- nasdaq100: 1 Symbole ohne Preisdaten (Beispiele: QQQ)
- nasdaq100: Benchmark/Universum konnte nicht geladen werden; Ergebnisse ausgelassen.

## Ergebnisübersicht (Universe/Split/Variante)
Keine auswertbaren Ergebnisse erzeugt.

## OOS-Bestätigung
Variante gilt nur als bestätigt, wenn beide OOS-Phasen die Akzeptanzregeln erfüllen.
Keine bestätigbaren Varianten.

## Sensitivitätsstabilität (In-Sample, Top-Nachbarn je Familie)
Keine Sensitivitätsdaten verfügbar.