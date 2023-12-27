pip install -r requirements.txt

# Compile and install pointnet2 operators.
cd pointnet2
python setup.py install

# Compile and install knn operator.
cd knn
python setup.py install

# install cpp extensions, the pointnet++ library
cd openpoints/cpp/pointnet2_batch
python setup.py install
cd ../

# grid_subsampling library. necessary only if interested in S3DIS_sphere
cd openpoints/cpp/subsampling
python setup.py build_ext --inplace
cd ..

# # point transformer library. Necessary only if interested in Point Transformer and Stratified Transformer
cd pointops/
python setup.py install
cd ..


# Blow are functions that optional. Necessary only if interested in reconstruction tasks such as completion
cd chamfer_dist
python setup.py install --user
cd ../emd
python setup.py install --user
cd ../../../
