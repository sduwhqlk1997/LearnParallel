matio安装方法：
wget https://github.com/tbeu/matio/releases/download/v1.5.25/matio-1.5.25.tar.gz
tar -xzf matio-1.5.25.tar.gz
cd matio-1.5.25

sudo apt update
sudo apt install zlib1g-dev libhdf5-dev

./configure \
--prefix=/home/qiu/qlk/Cppackage/matio \ & 记得修改安装路径
--with-hdf5 \
CPPFLAGS="-I/usr/include/hdf5/serial" \
LDFLAGS="-L/usr/lib/x86_64-linux-gnu/hdf5/serial"

make -j$(nproc)
make install