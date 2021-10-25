echo "Installing dependencies..."
pip install -r requirements.txt

function getData {
    cd datasets && \
    wget https://crisisnlp.qcri.org/data/crisismmd/CrisisMMD_v2.0.tar.gz && \ 
    tar -zxf CrisisMMD_v2.0.tar.gz && \
    cd CrisisMMD_v2.0 && \ 
    unzip crisismmd_datasplit_all.zip && \
    rm crisismmd_datasplit_all.zip && \
    cd .. && \
    rm CrisisMMD_v2.0.tar.gz && \
    cd ..
}

echo "Checking if CrisisMMD_v2.0 dataset exists..."
[ ! -d "datasets" ] && mkdir datasets
[ ! -d "datasets/CrisisMMD_v2.0" ] && getData



