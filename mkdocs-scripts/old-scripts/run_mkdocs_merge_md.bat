@echo off
setlocal enableextensions

echo Get the current directory
set "currentDir=%CD%"

echo Change the current working directory to the script directory
cd /d "%~dp0"

echo Change the working directory to the parent folder
cd ..


for /d %%i in (tex2pdf.*) do (
    echo deleting "%%~i"...
    rd /s /q "%%i"
    echo "%%~i" deleted.
)

echo All tex2pdf.* folders deleted.

echo Deleting mkdocs_***.md prefix files

rem Delete any remaining mkdocs_ folders
for /r %%# in (mkdocs_*) do (
    del "%%#"
)

echo Deleting files generated with mkdocs_***.md 

rem Delete any remaining mkdocs_ folders
for /r %%# in (mkdocs_*.slide.pdf mkdocs_*.slide.html mkdocs_*.slide.pptx mkdocs_*.word.pptx) do (
    del "%%#"
	echo "%%~i" deleted.
)

echo All mkdocs_* related files deleted, building again

java -jar markdown-slide-converter.jar --mergefolder docs

echo Revert to the original directory
cd "%currentDir%"

echo Build completed...

pause
@endlocal



