# Python venv, will have to change later for automatic workflow generation of tests
source _testsuite_venv/bin/activate

# Run all PyTorch scripts
for ((i = 1 ; i < 52 ; i++)); do
    echo "Test case $i"
    if [ $i -lt 10 ]; then
        path="./test_cases/net_import/00$i/create_testdata/net.py"
    else
        path="./test_cases/net_import/0$i/create_testdata/net.py"
    fi
    PYTHONPATH=. python3 $path
done

exit 0
