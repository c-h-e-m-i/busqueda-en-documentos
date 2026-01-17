@echo off
REM - Usamos codificación UTF-8:
set PYTHONUTF8=1
chcp 65001 >nul

REM - Habilitamos variables locales y creamos una para contar los tests ejecutados y otra para los pasados:
setlocal enabledelayedexpansion
set count=0
set passed=0

for %%I in (100 500) do (
  REM - Aumentamos en 1 el contador de tests totales:
  set /a count+=1
  echo.
  echo [!count!] Ejecutando: ALT_Searcher.py indices/index_%%I.bin -T referencias/result_%%I_original.txt
  
  REM - Guardamos la hora de inicio:
  set "start=!time!"
  
  REM - Ejecución de la tarea:
  python ALT_Searcher.py indices/index_%%I.bin -T referencias/result_%%I_original.txt > temp_out.txt
  
  REM - Guardamos la hora de fin:
  set "end=!time!"
  
  REM - Abrimos el archivo donde está la salida, y sacamos su última línea:
  type temp_out.txt
  for /f "usebackq delims=" %%L in ("temp_out.txt") do set last=%%L
  
  REM - Si es igual a "Parece que todo está bien, buen trabajo" (sin la , porque marca final de línea aquí),
  REM - sumamos 1 al número de tests pasados:
  if "!last!"=="Parece que todo está bien, buen trabajo" (
      set /a passed+=1
  )
  
  REM - Mostramos por consola las horas de inicio y fin:
  echo.
  echo Inicio: !start!
  echo Fin: !end!
  
  REM - Mostramos por consola el número de tests pasados frente a los ejecutados:
  echo.
  echo Tests pasados hasta ahora: !passed! de !count!
)

for %%D in (levenshtein levenshtein_o) do (
  for %%I in (100 500) do (
    for %%T in (1 2 3) do (
	  REM - Aumentamos en 1 el contador de tests totales:
	  set /a count+=1
	  echo.
	  echo [!count!] Ejecutando: ALT_Searcher.py indices/index_%%I.bin -T referencias/result_%%I_levenshtein_%%T.txt -s -t %%T -d "%%D"
	  
	  REM - Guardamos la hora de inicio:
      set "start=!time!"
  
	  REM - Ejecución de la tarea:
      python ALT_Searcher.py indices/index_%%I.bin -T referencias/result_%%I_levenshtein_%%T.txt -s -t %%T -d "%%D" > temp_out.txt
	  
	  REM - Guardamos la hora de fin:
	  set "end=!time!"
  
      REM - Abrimos el archivo donde está la salida, y sacamos su última línea:
	  type temp_out.txt
	  for /f "usebackq delims=" %%L in ("temp_out.txt") do set last=%%L

      REM - Si es igual a "Parece que todo está bien, buen trabajo" (sin la !, porque marca final de línea aquí),
      REM - sumamos 1 al número de tests pasados:
	  if "!last!"=="Parece que todo está bien, buen trabajo" (
        set /a passed+=1
      )
	  
	  REM - Mostramos por consola las horas de inicio y fin:
	  echo.
	  echo Inicio: !start!
	  echo Fin: !end!
	  
	  REM - Mostramos por consola el número de tests pasados frente a los ejecutados:
	  echo.
      echo Tests pasados hasta ahora: !passed! de !count!
    )
  )
)

for %%D in (damerau_r damerau_i) do (
  for %%I in (100 500) do (
    for %%T in (1 2 3) do (
	  REM - Aumentamos en 1 el contador de tests totales:
	  set /a count+=1
	  echo.
	  echo [!count!] Ejecutando: ALT_Searcher.py indices/index_%%I.bin -T referencias/result_%%I_%%D_%%T.txt -s -t %%T -d "%%D"
	  
	  REM - Guardamos la hora de inicio:
      set "start=!time!"
  
	  REM - Ejecución de la tarea:
      python ALT_Searcher.py indices/index_%%I.bin -T referencias/result_%%I_%%D_%%T.txt -s -t %%T -d "%%D" > temp_out.txt
	  
	  REM - Guardamos la hora de fin:
	  set "end=!time!"
  
      REM - Abrimos el archivo donde está la salida, y sacamos su última línea:
	  type temp_out.txt
	  for /f "usebackq delims=" %%L in ("temp_out.txt") do set last=%%L

      REM - Si es igual a "Parece que todo está bien, buen trabajo" (sin la !, porque marca final de línea aquí),
      REM - sumamos 1 al número de tests pasados:
	  if "!last!"=="Parece que todo está bien, buen trabajo" (
        set /a passed+=1
      )
	  
	  REM - Mostramos por consola las horas de inicio y fin:
	  echo.
	  echo Inicio: !start!
	  echo Fin: !end!
	  
	  REM - Mostramos por consola el número de tests pasados frente a los ejecutados:
	  echo.
      echo Tests pasados hasta ahora: !passed! de !count!
    )
  )
)