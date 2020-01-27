rm -f assignment2.zip 
zip -r assignment2.zip . -i "Problem_*.ipynb" "lib/*.py" -x "*.ipynb_checkpoints*"
