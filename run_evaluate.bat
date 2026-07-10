@echo off
REM ============================================================
REM Valutazione backbone su tutte le combinazioni:
REM   normalizzazione = standard / l2 / none
REM   pca-n-components = 0 (niente PCA) / 5 / 10 / 20
REM = 12 run totali, ognuna in una cartella dedicata.
REM Legge tutti i .npy in data/glomeruli/embeddings (5 CNN + 4 DINO).
REM Ogni run calcola sia Ward sia HDBSCAN (entrambi deterministici).
REM pca-n-components 0 = nessuna PCA (confronto per giustificare l'uso della PCA).
REM ============================================================

setlocal
set TF_CPP_MIN_LOG_LEVEL=3
set TF_ENABLE_ONEDNN_OPTS=1

set EMB=data/glomeruli/embeddings
set COMMON=--embeddings-dir %EMB% --hopkins-n-runs 100

for %%N in (standard l2 none) do (
  for %%P in (0 5 10 20) do (
    echo ============================================================
    echo Normalizzazione=%%N  PCA=%%P
    echo ============================================================
    if "%%P"=="0" (
      python scripts/evaluate_backbones.py %COMMON% --pca-n-components 0 --normalization %%N --output-dir results/eval_%%N_nopca
    ) else (
      python scripts/evaluate_backbones.py %COMMON% --pca-n-components %%P --normalization %%N --output-dir results/eval_%%N_pca%%P
    )
    if errorlevel 1 goto :error
  )
)

echo.
echo ############################################################
echo TUTTE LE 12 VALUTAZIONI COMPLETATE
echo ############################################################
echo Risultati nelle cartelle results/eval_<norm>_<pca>/
goto :end

:error
echo.
echo ############################################################
echo ERRORE durante una valutazione. Controlla il messaggio sopra.
echo ############################################################

:end
endlocal
pause