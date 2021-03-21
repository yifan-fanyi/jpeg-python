### JPEG in Python
### 2021.03.21
@Yifan  
    
***

1. I only find the luma huffman table, so that only the luma huffman table is used; I do not want to get the table from the flowchart in the ITU-t81;
```
https://digital-library.theiet.org/docserver/fulltext/books/te/pbte054e/PBTE054E_appendixb.pdf?expires=1616202373&id=id&accname=guest&checksum=53BFBEA9EB4B0800296C8841059E853B
```
2. The bit stream only include the DC and AC bits, no header file included;

3. The input block size can be any NXN, the corresponding quantization factor is generated from the original 8X8 quantization matrix using interploation;

***
```python
# example encoding
from encoder import eJPEG
eJ = eJPEG(Qf=10, N=8, grayscale=True)
stream = eJ.encode(name)

# example decoding
from decoder import dJPEG
dJ = dJPEG(H=512, W=512, Qf=10, N=8, grayscale=True)
iX = dJ.decode(stream)
```
 
