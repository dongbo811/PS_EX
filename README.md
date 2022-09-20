
# EX Results

| Models      | Parameters  | FLOPs         | PQ         | PQ_{things}        | PQ_{stuff}         | AP        | mIoU         | record         |
| :---        |    :----:   |      :----:   |      :----:   |      :----:   |      :----:  |     :----:   |      :----:    |     ---: |
| ex11 (dlc 8x) | 44M       | --   |     62.2         | 55.0        | 67.4  |  |78.2| 常规分离   |
| ex12 (dlc 8x) | 44M       | --   |     61.5         | 53.3        | 67.5  |   ||先实例后合并query   |
| ex13=7_1 (dlc 8x) | 46M       | --   |    62.0           | 54.0       | 67.8  ||| dab position add replaced the cat |
| ex14 (dlc 8x) | 44M       | 521G   |      62.2         |    54.8    |  67.8  |  38.4    |  79.4  |增加多尺度特征为6个尺度 |
| ex15 (dlc 8x) | 46M       | --   |     62.2         |    54.8    | 67.5 | -- | 79.4 | 多尺度特征反序 |
| ex16 (dlc 8x) | 46M       | --   |              |        |  || | gfnet代替deformable attention |
| ex17 (dlc 8x) | 44M       | 521G   |      61.3     |    53.0    |  67.4  |  --   |  --  |ex14+反序 |
| ex18 (dlc 8x) | 44M       | 521G   |      61.6         |    53.8    |  67.3  |  38.4    |  79.4  |ex17+decoder_layers = 7 |
| ex19 (dlc 8x) | 44M       | 521G   |      61.9        |    54.6    |  67.2  |  --    |  --  |ex14 + pe = edge from mask features |
| ex20 (dlc 8x) | 44M       | 521G   |      62.2         |    54.8    |  67.8  |  38.4    |  79.4  |ex14 + pe = edge from x[-1] |
| ex21 (dsw 8x) | 44M       | 521G   |      62.2         |    54.8    |  67.8  |  38.4    |  79.4  |ex14 + pe = edge from x[-2] |
| ex22 (dsw 8x) | 44M       | 521G   |      62.2         |    54.8    |  67.8  |  38.4    |  79.4  |ex14 + pe = edge from x[-3] |
| ex24 (our) | 44M       | 521G   |      62.5         |    56.1    |  67.1  |  38.4    |  79.4  | edge positional embedding |
