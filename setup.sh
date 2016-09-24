cd $( dirname "${BASH_SOURCE[0]}" )
echo `pwd`

virtualenv venv
source venv/bin/activate

pip install --upgrade pip
pip install numpy

SBIN_PATH1="$(cd lib && pwd)/?.lua"
SBIN_PATH2="$(cd lib && pwd)/?/init.lua"
SBIN_PATH3="$(cd lib && pwd)/?/?.lua"
SBIN_PATH="${SBIN_PATH1};${SBIN_PATH2};${SBIN_PATH3}"
echo -e "Appending $SBIN_PATH to LUA_PATH\n"
echo -e "\nexport LUA_PATH=\"$LUA_PATH;$SBIN_PATH\"" >> venv/bin/activate 
