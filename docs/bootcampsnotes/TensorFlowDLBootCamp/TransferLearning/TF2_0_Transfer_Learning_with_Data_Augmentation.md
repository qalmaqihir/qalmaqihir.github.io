================
by Jawad Haider

- <a href="#transfer-learning-with-data-augmentation"
  id="toc-transfer-learning-with-data-augmentation">Transfer Learning with
  Data Augmentation</a>

## Transfer Learning with Data Augmentation

``` python
# Install TensorFlow
# !pip install -q tensorflow-gpu==2.0.0-rc0

try:
  %tensorflow_version 2.x  # Colab only.
except Exception:
  pass

import tensorflow as tf
print(tf.__version__)
```

    `%tensorflow_version` only switches the major version: `1.x` or `2.x`.
    You set: `2.x  # Colab only.`. This will be interpreted as: `2.x`.


    TensorFlow 2.x selected.
    2.0.0

``` python
# More imports
from tensorflow.keras.layers import Input, Dense, Flatten
from tensorflow.keras.applications.vgg16 import VGG16 as PretrainedModel, \
  preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from glob import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys, os
```

``` python
# Data from: https://mmspg.epfl.ch/downloads/food-image-datasets/
# !wget --passive-ftp --prefer-family=ipv4 --ftp-user FoodImage@grebvm2.epfl.ch \
#  --ftp-password Cahc1moo -nc ftp://tremplin.epfl.ch/Food-5K.zip
!wget -nc https://lazyprogrammer.me/course_files/Food-5K.zip
```

    --2019-12-04 20:50:28--  https://lazyprogrammer.me/course_files/Food-5K.zip
    Resolving lazyprogrammer.me (lazyprogrammer.me)... 104.31.81.48, 104.31.80.48, 2606:4700:30::681f:5030, ...
    Connecting to lazyprogrammer.me (lazyprogrammer.me)|104.31.81.48|:443... connected.
    HTTP request sent, awaiting response... 200 OK
    Length: 447001986 (426M) [application/zip]
    Saving to: ‘Food-5K.zip’

    Food-5K.zip         100%[===================>] 426.29M  3.94MB/s    in 48s     

    2019-12-04 20:51:17 (8.87 MB/s) - ‘Food-5K.zip’ saved [447001986/447001986]

``` python
!unzip -qq -o Food-5K.zip
```

``` python
!ls
```

    Food-5K  Food-5K.zip  __MACOSX  sample_data

``` python
!ls Food-5K/training
```

    0_0.jpg     0_1387.jpg  0_422.jpg  0_809.jpg   1_1195.jpg  1_230.jpg  1_617.jpg
    0_1000.jpg  0_1388.jpg  0_423.jpg  0_80.jpg    1_1196.jpg  1_231.jpg  1_618.jpg
    0_1001.jpg  0_1389.jpg  0_424.jpg  0_810.jpg   1_1197.jpg  1_232.jpg  1_619.jpg
    0_1002.jpg  0_138.jpg   0_425.jpg  0_811.jpg   1_1198.jpg  1_233.jpg  1_61.jpg
    0_1003.jpg  0_1390.jpg  0_426.jpg  0_812.jpg   1_1199.jpg  1_234.jpg  1_620.jpg
    0_1004.jpg  0_1391.jpg  0_427.jpg  0_813.jpg   1_119.jpg   1_235.jpg  1_621.jpg
    0_1005.jpg  0_1392.jpg  0_428.jpg  0_814.jpg   1_11.jpg    1_236.jpg  1_622.jpg
    0_1006.jpg  0_1393.jpg  0_429.jpg  0_815.jpg   1_1200.jpg  1_237.jpg  1_623.jpg
    0_1007.jpg  0_1394.jpg  0_42.jpg   0_816.jpg   1_1201.jpg  1_238.jpg  1_624.jpg
    0_1008.jpg  0_1395.jpg  0_430.jpg  0_817.jpg   1_1202.jpg  1_239.jpg  1_625.jpg
    0_1009.jpg  0_1396.jpg  0_431.jpg  0_818.jpg   1_1203.jpg  1_23.jpg   1_626.jpg
    0_100.jpg   0_1397.jpg  0_432.jpg  0_819.jpg   1_1204.jpg  1_240.jpg  1_627.jpg
    0_1010.jpg  0_1398.jpg  0_433.jpg  0_81.jpg    1_1205.jpg  1_241.jpg  1_628.jpg
    0_1011.jpg  0_1399.jpg  0_434.jpg  0_820.jpg   1_1206.jpg  1_242.jpg  1_629.jpg
    0_1012.jpg  0_139.jpg   0_435.jpg  0_821.jpg   1_1207.jpg  1_243.jpg  1_62.jpg
    0_1013.jpg  0_13.jpg    0_436.jpg  0_822.jpg   1_1208.jpg  1_244.jpg  1_630.jpg
    0_1014.jpg  0_1400.jpg  0_437.jpg  0_823.jpg   1_1209.jpg  1_245.jpg  1_631.jpg
    0_1015.jpg  0_1401.jpg  0_438.jpg  0_824.jpg   1_120.jpg   1_246.jpg  1_632.jpg
    0_1016.jpg  0_1402.jpg  0_439.jpg  0_825.jpg   1_1210.jpg  1_247.jpg  1_633.jpg
    0_1017.jpg  0_1403.jpg  0_43.jpg   0_826.jpg   1_1211.jpg  1_248.jpg  1_634.jpg
    0_1018.jpg  0_1404.jpg  0_440.jpg  0_827.jpg   1_1212.jpg  1_249.jpg  1_635.jpg
    0_1019.jpg  0_1405.jpg  0_441.jpg  0_828.jpg   1_1213.jpg  1_24.jpg   1_636.jpg
    0_101.jpg   0_1406.jpg  0_442.jpg  0_829.jpg   1_1214.jpg  1_250.jpg  1_637.jpg
    0_1020.jpg  0_1407.jpg  0_443.jpg  0_82.jpg    1_1215.jpg  1_251.jpg  1_638.jpg
    0_1021.jpg  0_1408.jpg  0_444.jpg  0_830.jpg   1_1216.jpg  1_252.jpg  1_639.jpg
    0_1022.jpg  0_1409.jpg  0_445.jpg  0_831.jpg   1_1217.jpg  1_253.jpg  1_63.jpg
    0_1023.jpg  0_140.jpg   0_446.jpg  0_832.jpg   1_1218.jpg  1_254.jpg  1_640.jpg
    0_1024.jpg  0_1410.jpg  0_447.jpg  0_833.jpg   1_1219.jpg  1_255.jpg  1_641.jpg
    0_1025.jpg  0_1411.jpg  0_448.jpg  0_834.jpg   1_121.jpg   1_256.jpg  1_642.jpg
    0_1026.jpg  0_1412.jpg  0_449.jpg  0_835.jpg   1_1220.jpg  1_257.jpg  1_643.jpg
    0_1027.jpg  0_1413.jpg  0_44.jpg   0_836.jpg   1_1221.jpg  1_258.jpg  1_644.jpg
    0_1028.jpg  0_1414.jpg  0_450.jpg  0_837.jpg   1_1222.jpg  1_259.jpg  1_645.jpg
    0_1029.jpg  0_1415.jpg  0_451.jpg  0_838.jpg   1_1223.jpg  1_25.jpg   1_646.jpg
    0_102.jpg   0_1416.jpg  0_452.jpg  0_839.jpg   1_1224.jpg  1_260.jpg  1_647.jpg
    0_1030.jpg  0_1417.jpg  0_453.jpg  0_83.jpg    1_1225.jpg  1_261.jpg  1_648.jpg
    0_1031.jpg  0_1418.jpg  0_454.jpg  0_840.jpg   1_1226.jpg  1_262.jpg  1_649.jpg
    0_1032.jpg  0_1419.jpg  0_455.jpg  0_841.jpg   1_1227.jpg  1_263.jpg  1_64.jpg
    0_1033.jpg  0_141.jpg   0_456.jpg  0_842.jpg   1_1228.jpg  1_264.jpg  1_650.jpg
    0_1034.jpg  0_1420.jpg  0_457.jpg  0_843.jpg   1_1229.jpg  1_265.jpg  1_651.jpg
    0_1035.jpg  0_1421.jpg  0_458.jpg  0_844.jpg   1_122.jpg   1_266.jpg  1_652.jpg
    0_1036.jpg  0_1422.jpg  0_459.jpg  0_845.jpg   1_1230.jpg  1_267.jpg  1_653.jpg
    0_1037.jpg  0_1423.jpg  0_45.jpg   0_846.jpg   1_1231.jpg  1_268.jpg  1_654.jpg
    0_1038.jpg  0_1424.jpg  0_460.jpg  0_847.jpg   1_1232.jpg  1_269.jpg  1_655.jpg
    0_1039.jpg  0_1425.jpg  0_461.jpg  0_848.jpg   1_1233.jpg  1_26.jpg   1_656.jpg
    0_103.jpg   0_1426.jpg  0_462.jpg  0_849.jpg   1_1234.jpg  1_270.jpg  1_657.jpg
    0_1040.jpg  0_1427.jpg  0_463.jpg  0_84.jpg    1_1235.jpg  1_271.jpg  1_658.jpg
    0_1041.jpg  0_1428.jpg  0_464.jpg  0_850.jpg   1_1236.jpg  1_272.jpg  1_659.jpg
    0_1042.jpg  0_1429.jpg  0_465.jpg  0_851.jpg   1_1237.jpg  1_273.jpg  1_65.jpg
    0_1043.jpg  0_142.jpg   0_466.jpg  0_852.jpg   1_1238.jpg  1_274.jpg  1_660.jpg
    0_1044.jpg  0_1430.jpg  0_467.jpg  0_853.jpg   1_1239.jpg  1_275.jpg  1_661.jpg
    0_1045.jpg  0_1431.jpg  0_468.jpg  0_854.jpg   1_123.jpg   1_276.jpg  1_662.jpg
    0_1046.jpg  0_1432.jpg  0_469.jpg  0_855.jpg   1_1240.jpg  1_277.jpg  1_663.jpg
    0_1047.jpg  0_1433.jpg  0_46.jpg   0_856.jpg   1_1241.jpg  1_278.jpg  1_664.jpg
    0_1048.jpg  0_1434.jpg  0_470.jpg  0_857.jpg   1_1242.jpg  1_279.jpg  1_665.jpg
    0_1049.jpg  0_1435.jpg  0_471.jpg  0_858.jpg   1_1243.jpg  1_27.jpg   1_666.jpg
    0_104.jpg   0_1436.jpg  0_472.jpg  0_859.jpg   1_1244.jpg  1_280.jpg  1_667.jpg
    0_1050.jpg  0_1437.jpg  0_473.jpg  0_85.jpg    1_1245.jpg  1_281.jpg  1_668.jpg
    0_1051.jpg  0_1438.jpg  0_474.jpg  0_860.jpg   1_1246.jpg  1_282.jpg  1_669.jpg
    0_1052.jpg  0_1439.jpg  0_475.jpg  0_861.jpg   1_1247.jpg  1_283.jpg  1_66.jpg
    0_1053.jpg  0_143.jpg   0_476.jpg  0_862.jpg   1_1248.jpg  1_284.jpg  1_670.jpg
    0_1054.jpg  0_1440.jpg  0_477.jpg  0_863.jpg   1_1249.jpg  1_285.jpg  1_671.jpg
    0_1055.jpg  0_1441.jpg  0_478.jpg  0_864.jpg   1_124.jpg   1_286.jpg  1_672.jpg
    0_1056.jpg  0_1442.jpg  0_479.jpg  0_865.jpg   1_1250.jpg  1_287.jpg  1_673.jpg
    0_1057.jpg  0_1443.jpg  0_47.jpg   0_866.jpg   1_1251.jpg  1_288.jpg  1_674.jpg
    0_1058.jpg  0_1444.jpg  0_480.jpg  0_867.jpg   1_1252.jpg  1_289.jpg  1_675.jpg
    0_1059.jpg  0_1445.jpg  0_481.jpg  0_868.jpg   1_1253.jpg  1_28.jpg   1_676.jpg
    0_105.jpg   0_1446.jpg  0_482.jpg  0_869.jpg   1_1254.jpg  1_290.jpg  1_677.jpg
    0_1060.jpg  0_1447.jpg  0_483.jpg  0_86.jpg    1_1255.jpg  1_291.jpg  1_678.jpg
    0_1061.jpg  0_1448.jpg  0_484.jpg  0_870.jpg   1_1256.jpg  1_292.jpg  1_679.jpg
    0_1062.jpg  0_1449.jpg  0_485.jpg  0_871.jpg   1_1257.jpg  1_293.jpg  1_67.jpg
    0_1063.jpg  0_144.jpg   0_486.jpg  0_872.jpg   1_1258.jpg  1_294.jpg  1_680.jpg
    0_1064.jpg  0_1450.jpg  0_487.jpg  0_873.jpg   1_1259.jpg  1_295.jpg  1_681.jpg
    0_1065.jpg  0_1451.jpg  0_488.jpg  0_874.jpg   1_125.jpg   1_296.jpg  1_682.jpg
    0_1066.jpg  0_1452.jpg  0_489.jpg  0_875.jpg   1_1260.jpg  1_297.jpg  1_683.jpg
    0_1067.jpg  0_1453.jpg  0_48.jpg   0_876.jpg   1_1261.jpg  1_298.jpg  1_684.jpg
    0_1068.jpg  0_1454.jpg  0_490.jpg  0_877.jpg   1_1262.jpg  1_299.jpg  1_685.jpg
    0_1069.jpg  0_1455.jpg  0_491.jpg  0_878.jpg   1_1263.jpg  1_29.jpg   1_686.jpg
    0_106.jpg   0_1456.jpg  0_492.jpg  0_879.jpg   1_1264.jpg  1_2.jpg    1_687.jpg
    0_1070.jpg  0_1457.jpg  0_493.jpg  0_87.jpg    1_1265.jpg  1_300.jpg  1_688.jpg
    0_1071.jpg  0_1458.jpg  0_494.jpg  0_880.jpg   1_1266.jpg  1_301.jpg  1_689.jpg
    0_1072.jpg  0_1459.jpg  0_495.jpg  0_881.jpg   1_1267.jpg  1_302.jpg  1_68.jpg
    0_1073.jpg  0_145.jpg   0_496.jpg  0_882.jpg   1_1268.jpg  1_303.jpg  1_690.jpg
    0_1074.jpg  0_1460.jpg  0_497.jpg  0_883.jpg   1_1269.jpg  1_304.jpg  1_691.jpg
    0_1075.jpg  0_1461.jpg  0_498.jpg  0_884.jpg   1_126.jpg   1_305.jpg  1_692.jpg
    0_1076.jpg  0_1462.jpg  0_499.jpg  0_885.jpg   1_1270.jpg  1_306.jpg  1_693.jpg
    0_1077.jpg  0_1463.jpg  0_49.jpg   0_886.jpg   1_1271.jpg  1_307.jpg  1_694.jpg
    0_1078.jpg  0_1464.jpg  0_4.jpg    0_887.jpg   1_1272.jpg  1_308.jpg  1_695.jpg
    0_1079.jpg  0_1465.jpg  0_500.jpg  0_888.jpg   1_1273.jpg  1_309.jpg  1_696.jpg
    0_107.jpg   0_1466.jpg  0_501.jpg  0_889.jpg   1_1274.jpg  1_30.jpg   1_697.jpg
    0_1080.jpg  0_1467.jpg  0_502.jpg  0_88.jpg    1_1275.jpg  1_310.jpg  1_698.jpg
    0_1081.jpg  0_1468.jpg  0_503.jpg  0_890.jpg   1_1276.jpg  1_311.jpg  1_699.jpg
    0_1082.jpg  0_1469.jpg  0_504.jpg  0_891.jpg   1_1277.jpg  1_312.jpg  1_69.jpg
    0_1083.jpg  0_146.jpg   0_505.jpg  0_892.jpg   1_1278.jpg  1_313.jpg  1_6.jpg
    0_1084.jpg  0_1470.jpg  0_506.jpg  0_893.jpg   1_1279.jpg  1_314.jpg  1_700.jpg
    0_1085.jpg  0_1471.jpg  0_507.jpg  0_894.jpg   1_127.jpg   1_315.jpg  1_701.jpg
    0_1086.jpg  0_1472.jpg  0_508.jpg  0_895.jpg   1_1280.jpg  1_316.jpg  1_702.jpg
    0_1087.jpg  0_1473.jpg  0_509.jpg  0_896.jpg   1_1281.jpg  1_317.jpg  1_703.jpg
    0_1088.jpg  0_1474.jpg  0_50.jpg   0_897.jpg   1_1282.jpg  1_318.jpg  1_704.jpg
    0_1089.jpg  0_1475.jpg  0_510.jpg  0_898.jpg   1_1283.jpg  1_319.jpg  1_705.jpg
    0_108.jpg   0_1476.jpg  0_511.jpg  0_899.jpg   1_1284.jpg  1_31.jpg   1_706.jpg
    0_1090.jpg  0_1477.jpg  0_512.jpg  0_89.jpg    1_1285.jpg  1_320.jpg  1_707.jpg
    0_1091.jpg  0_1478.jpg  0_513.jpg  0_8.jpg     1_1286.jpg  1_321.jpg  1_708.jpg
    0_1092.jpg  0_1479.jpg  0_514.jpg  0_900.jpg   1_1287.jpg  1_322.jpg  1_709.jpg
    0_1093.jpg  0_147.jpg   0_515.jpg  0_901.jpg   1_1288.jpg  1_323.jpg  1_70.jpg
    0_1094.jpg  0_1480.jpg  0_516.jpg  0_902.jpg   1_1289.jpg  1_324.jpg  1_710.jpg
    0_1095.jpg  0_1481.jpg  0_517.jpg  0_903.jpg   1_128.jpg   1_325.jpg  1_711.jpg
    0_1096.jpg  0_1482.jpg  0_518.jpg  0_904.jpg   1_1290.jpg  1_326.jpg  1_712.jpg
    0_1097.jpg  0_1483.jpg  0_519.jpg  0_905.jpg   1_1291.jpg  1_327.jpg  1_713.jpg
    0_1098.jpg  0_1484.jpg  0_51.jpg   0_906.jpg   1_1292.jpg  1_328.jpg  1_714.jpg
    0_1099.jpg  0_1485.jpg  0_520.jpg  0_907.jpg   1_1293.jpg  1_329.jpg  1_715.jpg
    0_109.jpg   0_1486.jpg  0_521.jpg  0_908.jpg   1_1294.jpg  1_32.jpg   1_716.jpg
    0_10.jpg    0_1487.jpg  0_522.jpg  0_909.jpg   1_1295.jpg  1_330.jpg  1_717.jpg
    0_1100.jpg  0_1488.jpg  0_523.jpg  0_90.jpg    1_1296.jpg  1_331.jpg  1_718.jpg
    0_1101.jpg  0_1489.jpg  0_524.jpg  0_910.jpg   1_1297.jpg  1_332.jpg  1_719.jpg
    0_1102.jpg  0_148.jpg   0_525.jpg  0_911.jpg   1_1298.jpg  1_333.jpg  1_71.jpg
    0_1103.jpg  0_1490.jpg  0_526.jpg  0_912.jpg   1_1299.jpg  1_334.jpg  1_720.jpg
    0_1104.jpg  0_1491.jpg  0_527.jpg  0_913.jpg   1_129.jpg   1_335.jpg  1_721.jpg
    0_1105.jpg  0_1492.jpg  0_528.jpg  0_914.jpg   1_12.jpg    1_336.jpg  1_722.jpg
    0_1106.jpg  0_1493.jpg  0_529.jpg  0_915.jpg   1_1300.jpg  1_337.jpg  1_723.jpg
    0_1107.jpg  0_1494.jpg  0_52.jpg   0_916.jpg   1_1301.jpg  1_338.jpg  1_724.jpg
    0_1108.jpg  0_1495.jpg  0_530.jpg  0_917.jpg   1_1302.jpg  1_339.jpg  1_725.jpg
    0_1109.jpg  0_1496.jpg  0_531.jpg  0_918.jpg   1_1303.jpg  1_33.jpg   1_726.jpg
    0_110.jpg   0_1497.jpg  0_532.jpg  0_919.jpg   1_1304.jpg  1_340.jpg  1_727.jpg
    0_1110.jpg  0_1498.jpg  0_533.jpg  0_91.jpg    1_1305.jpg  1_341.jpg  1_728.jpg
    0_1111.jpg  0_1499.jpg  0_534.jpg  0_920.jpg   1_1306.jpg  1_342.jpg  1_729.jpg
    0_1112.jpg  0_149.jpg   0_535.jpg  0_921.jpg   1_1307.jpg  1_343.jpg  1_72.jpg
    0_1113.jpg  0_14.jpg    0_536.jpg  0_922.jpg   1_1308.jpg  1_344.jpg  1_730.jpg
    0_1114.jpg  0_150.jpg   0_537.jpg  0_923.jpg   1_1309.jpg  1_345.jpg  1_731.jpg
    0_1115.jpg  0_151.jpg   0_538.jpg  0_924.jpg   1_130.jpg   1_346.jpg  1_732.jpg
    0_1116.jpg  0_152.jpg   0_539.jpg  0_925.jpg   1_1310.jpg  1_347.jpg  1_733.jpg
    0_1117.jpg  0_153.jpg   0_53.jpg   0_926.jpg   1_1311.jpg  1_348.jpg  1_734.jpg
    0_1118.jpg  0_154.jpg   0_540.jpg  0_927.jpg   1_1312.jpg  1_349.jpg  1_735.jpg
    0_1119.jpg  0_155.jpg   0_541.jpg  0_928.jpg   1_1313.jpg  1_34.jpg   1_736.jpg
    0_111.jpg   0_156.jpg   0_542.jpg  0_929.jpg   1_1314.jpg  1_350.jpg  1_737.jpg
    0_1120.jpg  0_157.jpg   0_543.jpg  0_92.jpg    1_1315.jpg  1_351.jpg  1_738.jpg
    0_1121.jpg  0_158.jpg   0_544.jpg  0_930.jpg   1_1316.jpg  1_352.jpg  1_739.jpg
    0_1122.jpg  0_159.jpg   0_545.jpg  0_931.jpg   1_1317.jpg  1_353.jpg  1_73.jpg
    0_1123.jpg  0_15.jpg    0_546.jpg  0_932.jpg   1_1318.jpg  1_354.jpg  1_740.jpg
    0_1124.jpg  0_160.jpg   0_547.jpg  0_933.jpg   1_1319.jpg  1_355.jpg  1_741.jpg
    0_1125.jpg  0_161.jpg   0_548.jpg  0_934.jpg   1_131.jpg   1_356.jpg  1_742.jpg
    0_1126.jpg  0_162.jpg   0_549.jpg  0_935.jpg   1_1320.jpg  1_357.jpg  1_743.jpg
    0_1127.jpg  0_163.jpg   0_54.jpg   0_936.jpg   1_1321.jpg  1_358.jpg  1_744.jpg
    0_1128.jpg  0_164.jpg   0_550.jpg  0_937.jpg   1_1322.jpg  1_359.jpg  1_745.jpg
    0_1129.jpg  0_165.jpg   0_551.jpg  0_938.jpg   1_1323.jpg  1_35.jpg   1_746.jpg
    0_112.jpg   0_166.jpg   0_552.jpg  0_939.jpg   1_1324.jpg  1_360.jpg  1_747.jpg
    0_1130.jpg  0_167.jpg   0_553.jpg  0_93.jpg    1_1325.jpg  1_361.jpg  1_748.jpg
    0_1131.jpg  0_168.jpg   0_554.jpg  0_940.jpg   1_1326.jpg  1_362.jpg  1_749.jpg
    0_1132.jpg  0_169.jpg   0_555.jpg  0_941.jpg   1_1327.jpg  1_363.jpg  1_74.jpg
    0_1133.jpg  0_16.jpg    0_556.jpg  0_942.jpg   1_1328.jpg  1_364.jpg  1_750.jpg
    0_1134.jpg  0_170.jpg   0_557.jpg  0_943.jpg   1_1329.jpg  1_365.jpg  1_751.jpg
    0_1135.jpg  0_171.jpg   0_558.jpg  0_944.jpg   1_132.jpg   1_366.jpg  1_752.jpg
    0_1136.jpg  0_172.jpg   0_559.jpg  0_945.jpg   1_1330.jpg  1_367.jpg  1_753.jpg
    0_1137.jpg  0_173.jpg   0_55.jpg   0_946.jpg   1_1331.jpg  1_368.jpg  1_754.jpg
    0_1138.jpg  0_174.jpg   0_560.jpg  0_947.jpg   1_1332.jpg  1_369.jpg  1_755.jpg
    0_1139.jpg  0_175.jpg   0_561.jpg  0_948.jpg   1_1333.jpg  1_36.jpg   1_756.jpg
    0_113.jpg   0_176.jpg   0_562.jpg  0_949.jpg   1_1334.jpg  1_370.jpg  1_757.jpg
    0_1140.jpg  0_177.jpg   0_563.jpg  0_94.jpg    1_1335.jpg  1_371.jpg  1_758.jpg
    0_1141.jpg  0_178.jpg   0_564.jpg  0_950.jpg   1_1336.jpg  1_372.jpg  1_759.jpg
    0_1142.jpg  0_179.jpg   0_565.jpg  0_951.jpg   1_1337.jpg  1_373.jpg  1_75.jpg
    0_1143.jpg  0_17.jpg    0_566.jpg  0_952.jpg   1_1338.jpg  1_374.jpg  1_760.jpg
    0_1144.jpg  0_180.jpg   0_567.jpg  0_953.jpg   1_1339.jpg  1_375.jpg  1_761.jpg
    0_1145.jpg  0_181.jpg   0_568.jpg  0_954.jpg   1_133.jpg   1_376.jpg  1_762.jpg
    0_1146.jpg  0_182.jpg   0_569.jpg  0_955.jpg   1_1340.jpg  1_377.jpg  1_763.jpg
    0_1147.jpg  0_183.jpg   0_56.jpg   0_956.jpg   1_1341.jpg  1_378.jpg  1_764.jpg
    0_1148.jpg  0_184.jpg   0_570.jpg  0_957.jpg   1_1342.jpg  1_379.jpg  1_765.jpg
    0_1149.jpg  0_185.jpg   0_571.jpg  0_958.jpg   1_1343.jpg  1_37.jpg   1_766.jpg
    0_114.jpg   0_186.jpg   0_572.jpg  0_959.jpg   1_1344.jpg  1_380.jpg  1_767.jpg
    0_1150.jpg  0_187.jpg   0_573.jpg  0_95.jpg    1_1345.jpg  1_381.jpg  1_768.jpg
    0_1151.jpg  0_188.jpg   0_574.jpg  0_960.jpg   1_1346.jpg  1_382.jpg  1_769.jpg
    0_1152.jpg  0_189.jpg   0_575.jpg  0_961.jpg   1_1347.jpg  1_383.jpg  1_76.jpg
    0_1153.jpg  0_18.jpg    0_576.jpg  0_962.jpg   1_1348.jpg  1_384.jpg  1_770.jpg
    0_1154.jpg  0_190.jpg   0_577.jpg  0_963.jpg   1_1349.jpg  1_385.jpg  1_771.jpg
    0_1155.jpg  0_191.jpg   0_578.jpg  0_964.jpg   1_134.jpg   1_386.jpg  1_772.jpg
    0_1156.jpg  0_192.jpg   0_579.jpg  0_965.jpg   1_1350.jpg  1_387.jpg  1_773.jpg
    0_1157.jpg  0_193.jpg   0_57.jpg   0_966.jpg   1_1351.jpg  1_388.jpg  1_774.jpg
    0_1158.jpg  0_194.jpg   0_580.jpg  0_967.jpg   1_1352.jpg  1_389.jpg  1_775.jpg
    0_1159.jpg  0_195.jpg   0_581.jpg  0_968.jpg   1_1353.jpg  1_38.jpg   1_776.jpg
    0_115.jpg   0_196.jpg   0_582.jpg  0_969.jpg   1_1354.jpg  1_390.jpg  1_777.jpg
    0_1160.jpg  0_197.jpg   0_583.jpg  0_96.jpg    1_1355.jpg  1_391.jpg  1_778.jpg
    0_1161.jpg  0_198.jpg   0_584.jpg  0_970.jpg   1_1356.jpg  1_392.jpg  1_779.jpg
    0_1162.jpg  0_199.jpg   0_585.jpg  0_971.jpg   1_1357.jpg  1_393.jpg  1_77.jpg
    0_1163.jpg  0_19.jpg    0_586.jpg  0_972.jpg   1_1358.jpg  1_394.jpg  1_780.jpg
    0_1164.jpg  0_1.jpg 0_587.jpg  0_973.jpg   1_1359.jpg  1_395.jpg  1_781.jpg
    0_1165.jpg  0_200.jpg   0_588.jpg  0_974.jpg   1_135.jpg   1_396.jpg  1_782.jpg
    0_1166.jpg  0_201.jpg   0_589.jpg  0_975.jpg   1_1360.jpg  1_397.jpg  1_783.jpg
    0_1167.jpg  0_202.jpg   0_58.jpg   0_976.jpg   1_1361.jpg  1_398.jpg  1_784.jpg
    0_1168.jpg  0_203.jpg   0_590.jpg  0_977.jpg   1_1362.jpg  1_399.jpg  1_785.jpg
    0_1169.jpg  0_204.jpg   0_591.jpg  0_978.jpg   1_1363.jpg  1_39.jpg   1_786.jpg
    0_116.jpg   0_205.jpg   0_592.jpg  0_979.jpg   1_1364.jpg  1_3.jpg    1_787.jpg
    0_1170.jpg  0_206.jpg   0_593.jpg  0_97.jpg    1_1365.jpg  1_400.jpg  1_788.jpg
    0_1171.jpg  0_207.jpg   0_594.jpg  0_980.jpg   1_1366.jpg  1_401.jpg  1_789.jpg
    0_1172.jpg  0_208.jpg   0_595.jpg  0_981.jpg   1_1367.jpg  1_402.jpg  1_78.jpg
    0_1173.jpg  0_209.jpg   0_596.jpg  0_982.jpg   1_1368.jpg  1_403.jpg  1_790.jpg
    0_1174.jpg  0_20.jpg    0_597.jpg  0_983.jpg   1_1369.jpg  1_404.jpg  1_791.jpg
    0_1175.jpg  0_210.jpg   0_598.jpg  0_984.jpg   1_136.jpg   1_405.jpg  1_792.jpg
    0_1176.jpg  0_211.jpg   0_599.jpg  0_985.jpg   1_1370.jpg  1_406.jpg  1_793.jpg
    0_1177.jpg  0_212.jpg   0_59.jpg   0_986.jpg   1_1371.jpg  1_407.jpg  1_794.jpg
    0_1178.jpg  0_213.jpg   0_5.jpg    0_987.jpg   1_1372.jpg  1_408.jpg  1_795.jpg
    0_1179.jpg  0_214.jpg   0_600.jpg  0_988.jpg   1_1373.jpg  1_409.jpg  1_796.jpg
    0_117.jpg   0_215.jpg   0_601.jpg  0_989.jpg   1_1374.jpg  1_40.jpg   1_797.jpg
    0_1180.jpg  0_216.jpg   0_602.jpg  0_98.jpg    1_1375.jpg  1_410.jpg  1_798.jpg
    0_1181.jpg  0_217.jpg   0_603.jpg  0_990.jpg   1_1376.jpg  1_411.jpg  1_799.jpg
    0_1182.jpg  0_218.jpg   0_604.jpg  0_991.jpg   1_1377.jpg  1_412.jpg  1_79.jpg
    0_1183.jpg  0_219.jpg   0_605.jpg  0_992.jpg   1_1378.jpg  1_413.jpg  1_7.jpg
    0_1184.jpg  0_21.jpg    0_606.jpg  0_993.jpg   1_1379.jpg  1_414.jpg  1_800.jpg
    0_1185.jpg  0_220.jpg   0_607.jpg  0_994.jpg   1_137.jpg   1_415.jpg  1_801.jpg
    0_1186.jpg  0_221.jpg   0_608.jpg  0_995.jpg   1_1380.jpg  1_416.jpg  1_802.jpg
    0_1187.jpg  0_222.jpg   0_609.jpg  0_996.jpg   1_1381.jpg  1_417.jpg  1_803.jpg
    0_1188.jpg  0_223.jpg   0_60.jpg   0_997.jpg   1_1382.jpg  1_418.jpg  1_804.jpg
    0_1189.jpg  0_224.jpg   0_610.jpg  0_998.jpg   1_1383.jpg  1_419.jpg  1_805.jpg
    0_118.jpg   0_225.jpg   0_611.jpg  0_999.jpg   1_1384.jpg  1_41.jpg   1_806.jpg
    0_1190.jpg  0_226.jpg   0_612.jpg  0_99.jpg    1_1385.jpg  1_420.jpg  1_807.jpg
    0_1191.jpg  0_227.jpg   0_613.jpg  0_9.jpg     1_1386.jpg  1_421.jpg  1_808.jpg
    0_1192.jpg  0_228.jpg   0_614.jpg  1_0.jpg     1_1387.jpg  1_422.jpg  1_809.jpg
    0_1193.jpg  0_229.jpg   0_615.jpg  1_1000.jpg  1_1388.jpg  1_423.jpg  1_80.jpg
    0_1194.jpg  0_22.jpg    0_616.jpg  1_1001.jpg  1_1389.jpg  1_424.jpg  1_810.jpg
    0_1195.jpg  0_230.jpg   0_617.jpg  1_1002.jpg  1_138.jpg   1_425.jpg  1_811.jpg
    0_1196.jpg  0_231.jpg   0_618.jpg  1_1003.jpg  1_1390.jpg  1_426.jpg  1_812.jpg
    0_1197.jpg  0_232.jpg   0_619.jpg  1_1004.jpg  1_1391.jpg  1_427.jpg  1_813.jpg
    0_1198.jpg  0_233.jpg   0_61.jpg   1_1005.jpg  1_1392.jpg  1_428.jpg  1_814.jpg
    0_1199.jpg  0_234.jpg   0_620.jpg  1_1006.jpg  1_1393.jpg  1_429.jpg  1_815.jpg
    0_119.jpg   0_235.jpg   0_621.jpg  1_1007.jpg  1_1394.jpg  1_42.jpg   1_816.jpg
    0_11.jpg    0_236.jpg   0_622.jpg  1_1008.jpg  1_1395.jpg  1_430.jpg  1_817.jpg
    0_1200.jpg  0_237.jpg   0_623.jpg  1_1009.jpg  1_1396.jpg  1_431.jpg  1_818.jpg
    0_1201.jpg  0_238.jpg   0_624.jpg  1_100.jpg   1_1397.jpg  1_432.jpg  1_819.jpg
    0_1202.jpg  0_239.jpg   0_625.jpg  1_1010.jpg  1_1398.jpg  1_433.jpg  1_81.jpg
    0_1203.jpg  0_23.jpg    0_626.jpg  1_1011.jpg  1_1399.jpg  1_434.jpg  1_820.jpg
    0_1204.jpg  0_240.jpg   0_627.jpg  1_1012.jpg  1_139.jpg   1_435.jpg  1_821.jpg
    0_1205.jpg  0_241.jpg   0_628.jpg  1_1013.jpg  1_13.jpg    1_436.jpg  1_822.jpg
    0_1206.jpg  0_242.jpg   0_629.jpg  1_1014.jpg  1_1400.jpg  1_437.jpg  1_823.jpg
    0_1207.jpg  0_243.jpg   0_62.jpg   1_1015.jpg  1_1401.jpg  1_438.jpg  1_824.jpg
    0_1208.jpg  0_244.jpg   0_630.jpg  1_1016.jpg  1_1402.jpg  1_439.jpg  1_825.jpg
    0_1209.jpg  0_245.jpg   0_631.jpg  1_1017.jpg  1_1403.jpg  1_43.jpg   1_826.jpg
    0_120.jpg   0_246.jpg   0_632.jpg  1_1018.jpg  1_1404.jpg  1_440.jpg  1_827.jpg
    0_1210.jpg  0_247.jpg   0_633.jpg  1_1019.jpg  1_1405.jpg  1_441.jpg  1_828.jpg
    0_1211.jpg  0_248.jpg   0_634.jpg  1_101.jpg   1_1406.jpg  1_442.jpg  1_829.jpg
    0_1212.jpg  0_249.jpg   0_635.jpg  1_1020.jpg  1_1407.jpg  1_443.jpg  1_82.jpg
    0_1213.jpg  0_24.jpg    0_636.jpg  1_1021.jpg  1_1408.jpg  1_444.jpg  1_830.jpg
    0_1214.jpg  0_250.jpg   0_637.jpg  1_1022.jpg  1_1409.jpg  1_445.jpg  1_831.jpg
    0_1215.jpg  0_251.jpg   0_638.jpg  1_1023.jpg  1_140.jpg   1_446.jpg  1_832.jpg
    0_1216.jpg  0_252.jpg   0_639.jpg  1_1024.jpg  1_1410.jpg  1_447.jpg  1_833.jpg
    0_1217.jpg  0_253.jpg   0_63.jpg   1_1025.jpg  1_1411.jpg  1_448.jpg  1_834.jpg
    0_1218.jpg  0_254.jpg   0_640.jpg  1_1026.jpg  1_1412.jpg  1_449.jpg  1_835.jpg
    0_1219.jpg  0_255.jpg   0_641.jpg  1_1027.jpg  1_1413.jpg  1_44.jpg   1_836.jpg
    0_121.jpg   0_256.jpg   0_642.jpg  1_1028.jpg  1_1414.jpg  1_450.jpg  1_837.jpg
    0_1220.jpg  0_257.jpg   0_643.jpg  1_1029.jpg  1_1415.jpg  1_451.jpg  1_838.jpg
    0_1221.jpg  0_258.jpg   0_644.jpg  1_102.jpg   1_1416.jpg  1_452.jpg  1_839.jpg
    0_1222.jpg  0_259.jpg   0_645.jpg  1_1030.jpg  1_1417.jpg  1_453.jpg  1_83.jpg
    0_1223.jpg  0_25.jpg    0_646.jpg  1_1031.jpg  1_1418.jpg  1_454.jpg  1_840.jpg
    0_1224.jpg  0_260.jpg   0_647.jpg  1_1032.jpg  1_1419.jpg  1_455.jpg  1_841.jpg
    0_1225.jpg  0_261.jpg   0_648.jpg  1_1033.jpg  1_141.jpg   1_456.jpg  1_842.jpg
    0_1226.jpg  0_262.jpg   0_649.jpg  1_1034.jpg  1_1420.jpg  1_457.jpg  1_843.jpg
    0_1227.jpg  0_263.jpg   0_64.jpg   1_1035.jpg  1_1421.jpg  1_458.jpg  1_844.jpg
    0_1228.jpg  0_264.jpg   0_650.jpg  1_1036.jpg  1_1422.jpg  1_459.jpg  1_845.jpg
    0_1229.jpg  0_265.jpg   0_651.jpg  1_1037.jpg  1_1423.jpg  1_45.jpg   1_846.jpg
    0_122.jpg   0_266.jpg   0_652.jpg  1_1038.jpg  1_1424.jpg  1_460.jpg  1_847.jpg
    0_1230.jpg  0_267.jpg   0_653.jpg  1_1039.jpg  1_1425.jpg  1_461.jpg  1_848.jpg
    0_1231.jpg  0_268.jpg   0_654.jpg  1_103.jpg   1_1426.jpg  1_462.jpg  1_849.jpg
    0_1232.jpg  0_269.jpg   0_655.jpg  1_1040.jpg  1_1427.jpg  1_463.jpg  1_84.jpg
    0_1233.jpg  0_26.jpg    0_656.jpg  1_1041.jpg  1_1428.jpg  1_464.jpg  1_850.jpg
    0_1234.jpg  0_270.jpg   0_657.jpg  1_1042.jpg  1_1429.jpg  1_465.jpg  1_851.jpg
    0_1235.jpg  0_271.jpg   0_658.jpg  1_1043.jpg  1_142.jpg   1_466.jpg  1_852.jpg
    0_1236.jpg  0_272.jpg   0_659.jpg  1_1044.jpg  1_1430.jpg  1_467.jpg  1_853.jpg
    0_1237.jpg  0_273.jpg   0_65.jpg   1_1045.jpg  1_1431.jpg  1_468.jpg  1_854.jpg
    0_1238.jpg  0_274.jpg   0_660.jpg  1_1046.jpg  1_1432.jpg  1_469.jpg  1_855.jpg
    0_1239.jpg  0_275.jpg   0_661.jpg  1_1047.jpg  1_1433.jpg  1_46.jpg   1_856.jpg
    0_123.jpg   0_276.jpg   0_662.jpg  1_1048.jpg  1_1434.jpg  1_470.jpg  1_857.jpg
    0_1240.jpg  0_277.jpg   0_663.jpg  1_1049.jpg  1_1435.jpg  1_471.jpg  1_858.jpg
    0_1241.jpg  0_278.jpg   0_664.jpg  1_104.jpg   1_1436.jpg  1_472.jpg  1_859.jpg
    0_1242.jpg  0_279.jpg   0_665.jpg  1_1050.jpg  1_1437.jpg  1_473.jpg  1_85.jpg
    0_1243.jpg  0_27.jpg    0_666.jpg  1_1051.jpg  1_1438.jpg  1_474.jpg  1_860.jpg
    0_1244.jpg  0_280.jpg   0_667.jpg  1_1052.jpg  1_1439.jpg  1_475.jpg  1_861.jpg
    0_1245.jpg  0_281.jpg   0_668.jpg  1_1053.jpg  1_143.jpg   1_476.jpg  1_862.jpg
    0_1246.jpg  0_282.jpg   0_669.jpg  1_1054.jpg  1_1440.jpg  1_477.jpg  1_863.jpg
    0_1247.jpg  0_283.jpg   0_66.jpg   1_1055.jpg  1_1441.jpg  1_478.jpg  1_864.jpg
    0_1248.jpg  0_284.jpg   0_670.jpg  1_1056.jpg  1_1442.jpg  1_479.jpg  1_865.jpg
    0_1249.jpg  0_285.jpg   0_671.jpg  1_1057.jpg  1_1443.jpg  1_47.jpg   1_866.jpg
    0_124.jpg   0_286.jpg   0_672.jpg  1_1058.jpg  1_1444.jpg  1_480.jpg  1_867.jpg
    0_1250.jpg  0_287.jpg   0_673.jpg  1_1059.jpg  1_1445.jpg  1_481.jpg  1_868.jpg
    0_1251.jpg  0_288.jpg   0_674.jpg  1_105.jpg   1_1446.jpg  1_482.jpg  1_869.jpg
    0_1252.jpg  0_289.jpg   0_675.jpg  1_1060.jpg  1_1447.jpg  1_483.jpg  1_86.jpg
    0_1253.jpg  0_28.jpg    0_676.jpg  1_1061.jpg  1_1448.jpg  1_484.jpg  1_870.jpg
    0_1254.jpg  0_290.jpg   0_677.jpg  1_1062.jpg  1_1449.jpg  1_485.jpg  1_871.jpg
    0_1255.jpg  0_291.jpg   0_678.jpg  1_1063.jpg  1_144.jpg   1_486.jpg  1_872.jpg
    0_1256.jpg  0_292.jpg   0_679.jpg  1_1064.jpg  1_1450.jpg  1_487.jpg  1_873.jpg
    0_1257.jpg  0_293.jpg   0_67.jpg   1_1065.jpg  1_1451.jpg  1_488.jpg  1_874.jpg
    0_1258.jpg  0_294.jpg   0_680.jpg  1_1066.jpg  1_1452.jpg  1_489.jpg  1_875.jpg
    0_1259.jpg  0_295.jpg   0_681.jpg  1_1067.jpg  1_1453.jpg  1_48.jpg   1_876.jpg
    0_125.jpg   0_296.jpg   0_682.jpg  1_1068.jpg  1_1454.jpg  1_490.jpg  1_877.jpg
    0_1260.jpg  0_297.jpg   0_683.jpg  1_1069.jpg  1_1455.jpg  1_491.jpg  1_878.jpg
    0_1261.jpg  0_298.jpg   0_684.jpg  1_106.jpg   1_1456.jpg  1_492.jpg  1_879.jpg
    0_1262.jpg  0_299.jpg   0_685.jpg  1_1070.jpg  1_1457.jpg  1_493.jpg  1_87.jpg
    0_1263.jpg  0_29.jpg    0_686.jpg  1_1071.jpg  1_1458.jpg  1_494.jpg  1_880.jpg
    0_1264.jpg  0_2.jpg 0_687.jpg  1_1072.jpg  1_1459.jpg  1_495.jpg  1_881.jpg
    0_1265.jpg  0_300.jpg   0_688.jpg  1_1073.jpg  1_145.jpg   1_496.jpg  1_882.jpg
    0_1266.jpg  0_301.jpg   0_689.jpg  1_1074.jpg  1_1460.jpg  1_497.jpg  1_883.jpg
    0_1267.jpg  0_302.jpg   0_68.jpg   1_1075.jpg  1_1461.jpg  1_498.jpg  1_884.jpg
    0_1268.jpg  0_303.jpg   0_690.jpg  1_1076.jpg  1_1462.jpg  1_499.jpg  1_885.jpg
    0_1269.jpg  0_304.jpg   0_691.jpg  1_1077.jpg  1_1463.jpg  1_49.jpg   1_886.jpg
    0_126.jpg   0_305.jpg   0_692.jpg  1_1078.jpg  1_1464.jpg  1_4.jpg    1_887.jpg
    0_1270.jpg  0_306.jpg   0_693.jpg  1_1079.jpg  1_1465.jpg  1_500.jpg  1_888.jpg
    0_1271.jpg  0_307.jpg   0_694.jpg  1_107.jpg   1_1466.jpg  1_501.jpg  1_889.jpg
    0_1272.jpg  0_308.jpg   0_695.jpg  1_1080.jpg  1_1467.jpg  1_502.jpg  1_88.jpg
    0_1273.jpg  0_309.jpg   0_696.jpg  1_1081.jpg  1_1468.jpg  1_503.jpg  1_890.jpg
    0_1274.jpg  0_30.jpg    0_697.jpg  1_1082.jpg  1_1469.jpg  1_504.jpg  1_891.jpg
    0_1275.jpg  0_310.jpg   0_698.jpg  1_1083.jpg  1_146.jpg   1_505.jpg  1_892.jpg
    0_1276.jpg  0_311.jpg   0_699.jpg  1_1084.jpg  1_1470.jpg  1_506.jpg  1_893.jpg
    0_1277.jpg  0_312.jpg   0_69.jpg   1_1085.jpg  1_1471.jpg  1_507.jpg  1_894.jpg
    0_1278.jpg  0_313.jpg   0_6.jpg    1_1086.jpg  1_1472.jpg  1_508.jpg  1_895.jpg
    0_1279.jpg  0_314.jpg   0_700.jpg  1_1087.jpg  1_1473.jpg  1_509.jpg  1_896.jpg
    0_127.jpg   0_315.jpg   0_701.jpg  1_1088.jpg  1_1474.jpg  1_50.jpg   1_897.jpg
    0_1280.jpg  0_316.jpg   0_702.jpg  1_1089.jpg  1_1475.jpg  1_510.jpg  1_898.jpg
    0_1281.jpg  0_317.jpg   0_703.jpg  1_108.jpg   1_1476.jpg  1_511.jpg  1_899.jpg
    0_1282.jpg  0_318.jpg   0_704.jpg  1_1090.jpg  1_1477.jpg  1_512.jpg  1_89.jpg
    0_1283.jpg  0_319.jpg   0_705.jpg  1_1091.jpg  1_1478.jpg  1_513.jpg  1_8.jpg
    0_1284.jpg  0_31.jpg    0_706.jpg  1_1092.jpg  1_1479.jpg  1_514.jpg  1_900.jpg
    0_1285.jpg  0_320.jpg   0_707.jpg  1_1093.jpg  1_147.jpg   1_515.jpg  1_901.jpg
    0_1286.jpg  0_321.jpg   0_708.jpg  1_1094.jpg  1_1480.jpg  1_516.jpg  1_902.jpg
    0_1287.jpg  0_322.jpg   0_709.jpg  1_1095.jpg  1_1481.jpg  1_517.jpg  1_903.jpg
    0_1288.jpg  0_323.jpg   0_70.jpg   1_1096.jpg  1_1482.jpg  1_518.jpg  1_904.jpg
    0_1289.jpg  0_324.jpg   0_710.jpg  1_1097.jpg  1_1483.jpg  1_519.jpg  1_905.jpg
    0_128.jpg   0_325.jpg   0_711.jpg  1_1098.jpg  1_1484.jpg  1_51.jpg   1_906.jpg
    0_1290.jpg  0_326.jpg   0_712.jpg  1_1099.jpg  1_1485.jpg  1_520.jpg  1_907.jpg
    0_1291.jpg  0_327.jpg   0_713.jpg  1_109.jpg   1_1486.jpg  1_521.jpg  1_908.jpg
    0_1292.jpg  0_328.jpg   0_714.jpg  1_10.jpg    1_1487.jpg  1_522.jpg  1_909.jpg
    0_1293.jpg  0_329.jpg   0_715.jpg  1_1100.jpg  1_1488.jpg  1_523.jpg  1_90.jpg
    0_1294.jpg  0_32.jpg    0_716.jpg  1_1101.jpg  1_1489.jpg  1_524.jpg  1_910.jpg
    0_1295.jpg  0_330.jpg   0_717.jpg  1_1102.jpg  1_148.jpg   1_525.jpg  1_911.jpg
    0_1296.jpg  0_331.jpg   0_718.jpg  1_1103.jpg  1_1490.jpg  1_526.jpg  1_912.jpg
    0_1297.jpg  0_332.jpg   0_719.jpg  1_1104.jpg  1_1491.jpg  1_527.jpg  1_913.jpg
    0_1298.jpg  0_333.jpg   0_71.jpg   1_1105.jpg  1_1492.jpg  1_528.jpg  1_914.jpg
    0_1299.jpg  0_334.jpg   0_720.jpg  1_1106.jpg  1_1493.jpg  1_529.jpg  1_915.jpg
    0_129.jpg   0_335.jpg   0_721.jpg  1_1107.jpg  1_1494.jpg  1_52.jpg   1_916.jpg
    0_12.jpg    0_336.jpg   0_722.jpg  1_1108.jpg  1_1495.jpg  1_530.jpg  1_917.jpg
    0_1300.jpg  0_337.jpg   0_723.jpg  1_1109.jpg  1_1496.jpg  1_531.jpg  1_918.jpg
    0_1301.jpg  0_338.jpg   0_724.jpg  1_110.jpg   1_1497.jpg  1_532.jpg  1_919.jpg
    0_1302.jpg  0_339.jpg   0_725.jpg  1_1110.jpg  1_1498.jpg  1_533.jpg  1_91.jpg
    0_1303.jpg  0_33.jpg    0_726.jpg  1_1111.jpg  1_1499.jpg  1_534.jpg  1_920.jpg
    0_1304.jpg  0_340.jpg   0_727.jpg  1_1112.jpg  1_149.jpg   1_535.jpg  1_921.jpg
    0_1305.jpg  0_341.jpg   0_728.jpg  1_1113.jpg  1_14.jpg    1_536.jpg  1_922.jpg
    0_1306.jpg  0_342.jpg   0_729.jpg  1_1114.jpg  1_150.jpg   1_537.jpg  1_923.jpg
    0_1307.jpg  0_343.jpg   0_72.jpg   1_1115.jpg  1_151.jpg   1_538.jpg  1_924.jpg
    0_1308.jpg  0_344.jpg   0_730.jpg  1_1116.jpg  1_152.jpg   1_539.jpg  1_925.jpg
    0_1309.jpg  0_345.jpg   0_731.jpg  1_1117.jpg  1_153.jpg   1_53.jpg   1_926.jpg
    0_130.jpg   0_346.jpg   0_732.jpg  1_1118.jpg  1_154.jpg   1_540.jpg  1_927.jpg
    0_1310.jpg  0_347.jpg   0_733.jpg  1_1119.jpg  1_155.jpg   1_541.jpg  1_928.jpg
    0_1311.jpg  0_348.jpg   0_734.jpg  1_111.jpg   1_156.jpg   1_542.jpg  1_929.jpg
    0_1312.jpg  0_349.jpg   0_735.jpg  1_1120.jpg  1_157.jpg   1_543.jpg  1_92.jpg
    0_1313.jpg  0_34.jpg    0_736.jpg  1_1121.jpg  1_158.jpg   1_544.jpg  1_930.jpg
    0_1314.jpg  0_350.jpg   0_737.jpg  1_1122.jpg  1_159.jpg   1_545.jpg  1_931.jpg
    0_1315.jpg  0_351.jpg   0_738.jpg  1_1123.jpg  1_15.jpg    1_546.jpg  1_932.jpg
    0_1316.jpg  0_352.jpg   0_739.jpg  1_1124.jpg  1_160.jpg   1_547.jpg  1_933.jpg
    0_1317.jpg  0_353.jpg   0_73.jpg   1_1125.jpg  1_161.jpg   1_548.jpg  1_934.jpg
    0_1318.jpg  0_354.jpg   0_740.jpg  1_1126.jpg  1_162.jpg   1_549.jpg  1_935.jpg
    0_1319.jpg  0_355.jpg   0_741.jpg  1_1127.jpg  1_163.jpg   1_54.jpg   1_936.jpg
    0_131.jpg   0_356.jpg   0_742.jpg  1_1128.jpg  1_164.jpg   1_550.jpg  1_937.jpg
    0_1320.jpg  0_357.jpg   0_743.jpg  1_1129.jpg  1_165.jpg   1_551.jpg  1_938.jpg
    0_1321.jpg  0_358.jpg   0_744.jpg  1_112.jpg   1_166.jpg   1_552.jpg  1_939.jpg
    0_1322.jpg  0_359.jpg   0_745.jpg  1_1130.jpg  1_167.jpg   1_553.jpg  1_93.jpg
    0_1323.jpg  0_35.jpg    0_746.jpg  1_1131.jpg  1_168.jpg   1_554.jpg  1_940.jpg
    0_1324.jpg  0_360.jpg   0_747.jpg  1_1132.jpg  1_169.jpg   1_555.jpg  1_941.jpg
    0_1325.jpg  0_361.jpg   0_748.jpg  1_1133.jpg  1_16.jpg    1_556.jpg  1_942.jpg
    0_1326.jpg  0_362.jpg   0_749.jpg  1_1134.jpg  1_170.jpg   1_557.jpg  1_943.jpg
    0_1327.jpg  0_363.jpg   0_74.jpg   1_1135.jpg  1_171.jpg   1_558.jpg  1_944.jpg
    0_1328.jpg  0_364.jpg   0_750.jpg  1_1136.jpg  1_172.jpg   1_559.jpg  1_945.jpg
    0_1329.jpg  0_365.jpg   0_751.jpg  1_1137.jpg  1_173.jpg   1_55.jpg   1_946.jpg
    0_132.jpg   0_366.jpg   0_752.jpg  1_1138.jpg  1_174.jpg   1_560.jpg  1_947.jpg
    0_1330.jpg  0_367.jpg   0_753.jpg  1_1139.jpg  1_175.jpg   1_561.jpg  1_948.jpg
    0_1331.jpg  0_368.jpg   0_754.jpg  1_113.jpg   1_176.jpg   1_562.jpg  1_949.jpg
    0_1332.jpg  0_369.jpg   0_755.jpg  1_1140.jpg  1_177.jpg   1_563.jpg  1_94.jpg
    0_1333.jpg  0_36.jpg    0_756.jpg  1_1141.jpg  1_178.jpg   1_564.jpg  1_950.jpg
    0_1334.jpg  0_370.jpg   0_757.jpg  1_1142.jpg  1_179.jpg   1_565.jpg  1_951.jpg
    0_1335.jpg  0_371.jpg   0_758.jpg  1_1143.jpg  1_17.jpg    1_566.jpg  1_952.jpg
    0_1336.jpg  0_372.jpg   0_759.jpg  1_1144.jpg  1_180.jpg   1_567.jpg  1_953.jpg
    0_1337.jpg  0_373.jpg   0_75.jpg   1_1145.jpg  1_181.jpg   1_568.jpg  1_954.jpg
    0_1338.jpg  0_374.jpg   0_760.jpg  1_1146.jpg  1_182.jpg   1_569.jpg  1_955.jpg
    0_1339.jpg  0_375.jpg   0_761.jpg  1_1147.jpg  1_183.jpg   1_56.jpg   1_956.jpg
    0_133.jpg   0_376.jpg   0_762.jpg  1_1148.jpg  1_184.jpg   1_570.jpg  1_957.jpg
    0_1340.jpg  0_377.jpg   0_763.jpg  1_1149.jpg  1_185.jpg   1_571.jpg  1_958.jpg
    0_1341.jpg  0_378.jpg   0_764.jpg  1_114.jpg   1_186.jpg   1_572.jpg  1_959.jpg
    0_1342.jpg  0_379.jpg   0_765.jpg  1_1150.jpg  1_187.jpg   1_573.jpg  1_95.jpg
    0_1343.jpg  0_37.jpg    0_766.jpg  1_1151.jpg  1_188.jpg   1_574.jpg  1_960.jpg
    0_1344.jpg  0_380.jpg   0_767.jpg  1_1152.jpg  1_189.jpg   1_575.jpg  1_961.jpg
    0_1345.jpg  0_381.jpg   0_768.jpg  1_1153.jpg  1_18.jpg    1_576.jpg  1_962.jpg
    0_1346.jpg  0_382.jpg   0_769.jpg  1_1154.jpg  1_190.jpg   1_577.jpg  1_963.jpg
    0_1347.jpg  0_383.jpg   0_76.jpg   1_1155.jpg  1_191.jpg   1_578.jpg  1_964.jpg
    0_1348.jpg  0_384.jpg   0_770.jpg  1_1156.jpg  1_192.jpg   1_579.jpg  1_965.jpg
    0_1349.jpg  0_385.jpg   0_771.jpg  1_1157.jpg  1_193.jpg   1_57.jpg   1_966.jpg
    0_134.jpg   0_386.jpg   0_772.jpg  1_1158.jpg  1_194.jpg   1_580.jpg  1_967.jpg
    0_1350.jpg  0_387.jpg   0_773.jpg  1_1159.jpg  1_195.jpg   1_581.jpg  1_968.jpg
    0_1351.jpg  0_388.jpg   0_774.jpg  1_115.jpg   1_196.jpg   1_582.jpg  1_969.jpg
    0_1352.jpg  0_389.jpg   0_775.jpg  1_1160.jpg  1_197.jpg   1_583.jpg  1_96.jpg
    0_1353.jpg  0_38.jpg    0_776.jpg  1_1161.jpg  1_198.jpg   1_584.jpg  1_970.jpg
    0_1354.jpg  0_390.jpg   0_777.jpg  1_1162.jpg  1_199.jpg   1_585.jpg  1_971.jpg
    0_1355.jpg  0_391.jpg   0_778.jpg  1_1163.jpg  1_19.jpg    1_586.jpg  1_972.jpg
    0_1356.jpg  0_392.jpg   0_779.jpg  1_1164.jpg  1_1.jpg     1_587.jpg  1_973.jpg
    0_1357.jpg  0_393.jpg   0_77.jpg   1_1165.jpg  1_200.jpg   1_588.jpg  1_974.jpg
    0_1358.jpg  0_394.jpg   0_780.jpg  1_1166.jpg  1_201.jpg   1_589.jpg  1_975.jpg
    0_1359.jpg  0_395.jpg   0_781.jpg  1_1167.jpg  1_202.jpg   1_58.jpg   1_976.jpg
    0_135.jpg   0_396.jpg   0_782.jpg  1_1168.jpg  1_203.jpg   1_590.jpg  1_977.jpg
    0_1360.jpg  0_397.jpg   0_783.jpg  1_1169.jpg  1_204.jpg   1_591.jpg  1_978.jpg
    0_1361.jpg  0_398.jpg   0_784.jpg  1_116.jpg   1_205.jpg   1_592.jpg  1_979.jpg
    0_1362.jpg  0_399.jpg   0_785.jpg  1_1170.jpg  1_206.jpg   1_593.jpg  1_97.jpg
    0_1363.jpg  0_39.jpg    0_786.jpg  1_1171.jpg  1_207.jpg   1_594.jpg  1_980.jpg
    0_1364.jpg  0_3.jpg 0_787.jpg  1_1172.jpg  1_208.jpg   1_595.jpg  1_981.jpg
    0_1365.jpg  0_400.jpg   0_788.jpg  1_1173.jpg  1_209.jpg   1_596.jpg  1_982.jpg
    0_1366.jpg  0_401.jpg   0_789.jpg  1_1174.jpg  1_20.jpg    1_597.jpg  1_983.jpg
    0_1367.jpg  0_402.jpg   0_78.jpg   1_1175.jpg  1_210.jpg   1_598.jpg  1_984.jpg
    0_1368.jpg  0_403.jpg   0_790.jpg  1_1176.jpg  1_211.jpg   1_599.jpg  1_985.jpg
    0_1369.jpg  0_404.jpg   0_791.jpg  1_1177.jpg  1_212.jpg   1_59.jpg   1_986.jpg
    0_136.jpg   0_405.jpg   0_792.jpg  1_1178.jpg  1_213.jpg   1_5.jpg    1_987.jpg
    0_1370.jpg  0_406.jpg   0_793.jpg  1_1179.jpg  1_214.jpg   1_600.jpg  1_988.jpg
    0_1371.jpg  0_407.jpg   0_794.jpg  1_117.jpg   1_215.jpg   1_601.jpg  1_989.jpg
    0_1372.jpg  0_408.jpg   0_795.jpg  1_1180.jpg  1_216.jpg   1_602.jpg  1_98.jpg
    0_1373.jpg  0_409.jpg   0_796.jpg  1_1181.jpg  1_217.jpg   1_603.jpg  1_990.jpg
    0_1374.jpg  0_40.jpg    0_797.jpg  1_1182.jpg  1_218.jpg   1_604.jpg  1_991.jpg
    0_1375.jpg  0_410.jpg   0_798.jpg  1_1183.jpg  1_219.jpg   1_605.jpg  1_992.jpg
    0_1376.jpg  0_411.jpg   0_799.jpg  1_1184.jpg  1_21.jpg    1_606.jpg  1_993.jpg
    0_1377.jpg  0_412.jpg   0_79.jpg   1_1185.jpg  1_220.jpg   1_607.jpg  1_994.jpg
    0_1378.jpg  0_413.jpg   0_7.jpg    1_1186.jpg  1_221.jpg   1_608.jpg  1_995.jpg
    0_1379.jpg  0_414.jpg   0_800.jpg  1_1187.jpg  1_222.jpg   1_609.jpg  1_996.jpg
    0_137.jpg   0_415.jpg   0_801.jpg  1_1188.jpg  1_223.jpg   1_60.jpg   1_997.jpg
    0_1380.jpg  0_416.jpg   0_802.jpg  1_1189.jpg  1_224.jpg   1_610.jpg  1_998.jpg
    0_1381.jpg  0_417.jpg   0_803.jpg  1_118.jpg   1_225.jpg   1_611.jpg  1_999.jpg
    0_1382.jpg  0_418.jpg   0_804.jpg  1_1190.jpg  1_226.jpg   1_612.jpg  1_99.jpg
    0_1383.jpg  0_419.jpg   0_805.jpg  1_1191.jpg  1_227.jpg   1_613.jpg  1_9.jpg
    0_1384.jpg  0_41.jpg    0_806.jpg  1_1192.jpg  1_228.jpg   1_614.jpg
    0_1385.jpg  0_420.jpg   0_807.jpg  1_1193.jpg  1_229.jpg   1_615.jpg
    0_1386.jpg  0_421.jpg   0_808.jpg  1_1194.jpg  1_22.jpg    1_616.jpg

``` python
!mv Food-5K/* .
```

``` python
# look at an image for fun
plt.imshow(image.load_img('training/0_808.jpg'))
plt.show()
```

![](TF2_0_Transfer_Learning_with_Data_Augmentation_files/figure-gfm/cell-9-output-1.png)

``` python
# Food images start with 1, non-food images start with 0
plt.imshow(image.load_img('training/1_616.jpg'))
plt.show()
```

![](TF2_0_Transfer_Learning_with_Data_Augmentation_files/figure-gfm/cell-10-output-1.png)

``` python
!mkdir data
```

``` python
# Make directories to store the data Keras-style
!mkdir data/train
!mkdir data/test
!mkdir data/train/nonfood
!mkdir data/train/food
!mkdir data/test/nonfood
!mkdir data/test/food
```

``` python
# Move the images
# Note: we will consider 'training' to be the train set
#       'validation' folder will be the test set
#       ignore the 'evaluation' set
!mv training/0*.jpg data/train/nonfood
!mv training/1*.jpg data/train/food
!mv validation/0*.jpg data/test/nonfood
!mv validation/1*.jpg data/test/food
```

``` python
train_path = 'data/train'
valid_path = 'data/test'
```

``` python
# These images are pretty big and of different sizes
# Let's load them all in as the same (smaller) size
IMAGE_SIZE = [200, 200]
```

``` python
# useful for getting number of files
image_files = glob(train_path + '/*/*.jpg')
valid_image_files = glob(valid_path + '/*/*.jpg')
```

``` python
# useful for getting number of classes
folders = glob(train_path + '/*')
folders
```

    ['data/train/nonfood', 'data/train/food']

``` python
# look at an image for fun
plt.imshow(image.load_img(np.random.choice(image_files)))
plt.show()
```

![](TF2_0_Transfer_Learning_with_Data_Augmentation_files/figure-gfm/cell-18-output-1.png)

``` python
ptm = PretrainedModel(
    input_shape=IMAGE_SIZE + [3],
    weights='imagenet',
    include_top=False)
```

    Downloading data from https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5
    58892288/58889256 [==============================] - 6s 0us/step

``` python
# freeze pretrained model weights
ptm.trainable = False
```

``` python
# map the data into feature vectors

# Keras image data generator returns classes one-hot encoded

K = len(folders) # number of classes
x = Flatten()(ptm.output)
x = Dense(K, activation='softmax')(x)
```

``` python
# create a model object
model = Model(inputs=ptm.input, outputs=x)
```

``` python
# view the structure of the model
model.summary()
```

    Model: "model"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    input_1 (InputLayer)         [(None, 200, 200, 3)]     0         
    _________________________________________________________________
    block1_conv1 (Conv2D)        (None, 200, 200, 64)      1792      
    _________________________________________________________________
    block1_conv2 (Conv2D)        (None, 200, 200, 64)      36928     
    _________________________________________________________________
    block1_pool (MaxPooling2D)   (None, 100, 100, 64)      0         
    _________________________________________________________________
    block2_conv1 (Conv2D)        (None, 100, 100, 128)     73856     
    _________________________________________________________________
    block2_conv2 (Conv2D)        (None, 100, 100, 128)     147584    
    _________________________________________________________________
    block2_pool (MaxPooling2D)   (None, 50, 50, 128)       0         
    _________________________________________________________________
    block3_conv1 (Conv2D)        (None, 50, 50, 256)       295168    
    _________________________________________________________________
    block3_conv2 (Conv2D)        (None, 50, 50, 256)       590080    
    _________________________________________________________________
    block3_conv3 (Conv2D)        (None, 50, 50, 256)       590080    
    _________________________________________________________________
    block3_pool (MaxPooling2D)   (None, 25, 25, 256)       0         
    _________________________________________________________________
    block4_conv1 (Conv2D)        (None, 25, 25, 512)       1180160   
    _________________________________________________________________
    block4_conv2 (Conv2D)        (None, 25, 25, 512)       2359808   
    _________________________________________________________________
    block4_conv3 (Conv2D)        (None, 25, 25, 512)       2359808   
    _________________________________________________________________
    block4_pool (MaxPooling2D)   (None, 12, 12, 512)       0         
    _________________________________________________________________
    block5_conv1 (Conv2D)        (None, 12, 12, 512)       2359808   
    _________________________________________________________________
    block5_conv2 (Conv2D)        (None, 12, 12, 512)       2359808   
    _________________________________________________________________
    block5_conv3 (Conv2D)        (None, 12, 12, 512)       2359808   
    _________________________________________________________________
    block5_pool (MaxPooling2D)   (None, 6, 6, 512)         0         
    _________________________________________________________________
    flatten (Flatten)            (None, 18432)             0         
    _________________________________________________________________
    dense (Dense)                (None, 2)                 36866     
    =================================================================
    Total params: 14,751,554
    Trainable params: 36,866
    Non-trainable params: 14,714,688
    _________________________________________________________________

``` python
# create an instance of ImageDataGenerator
gen_train = ImageDataGenerator(
  rotation_range=20,
  width_shift_range=0.1,
  height_shift_range=0.1,
  shear_range=0.1,
  zoom_range=0.2,
  horizontal_flip=True,
  preprocessing_function=preprocess_input
)

gen_test = ImageDataGenerator(
  preprocessing_function=preprocess_input
)
```

``` python
batch_size = 128

# create generators
train_generator = gen_train.flow_from_directory(
  train_path,
  shuffle=True,
  target_size=IMAGE_SIZE,
  batch_size=batch_size,
)
valid_generator = gen_test.flow_from_directory(
  valid_path,
  target_size=IMAGE_SIZE,
  batch_size=batch_size,
)
```

    Found 3000 images belonging to 2 classes.
    Found 1000 images belonging to 2 classes.

``` python
model.compile(
  loss='categorical_crossentropy',
  optimizer='adam',
  metrics=['accuracy']
)
```

``` python
# fit the model
r = model.fit_generator(
  train_generator,
  validation_data=valid_generator,
  epochs=10,
  steps_per_epoch=int(np.ceil(len(image_files) / batch_size)),
  validation_steps=int(np.ceil(len(valid_image_files) / batch_size)),
)
```

    Epoch 1/10
    WARNING:tensorflow:From /tensorflow-2.0.0-rc0/python3.6/tensorflow_core/python/ops/math_grad.py:1394: where (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
    Instructions for updating:
    Use tf.where in 2.0, which has the same broadcast rule as np.where
    24/24 [==============================] - 145s 6s/step - loss: 0.9769 - accuracy: 0.9183 - val_loss: 0.2264 - val_accuracy: 0.9790
    Epoch 2/10
    24/24 [==============================] - 103s 4s/step - loss: 0.4751 - accuracy: 0.9643 - val_loss: 0.2551 - val_accuracy: 0.9780
    Epoch 3/10
    24/24 [==============================] - 104s 4s/step - loss: 0.4594 - accuracy: 0.9667 - val_loss: 0.3841 - val_accuracy: 0.9730
    Epoch 4/10
    24/24 [==============================] - 104s 4s/step - loss: 0.3127 - accuracy: 0.9773 - val_loss: 0.2724 - val_accuracy: 0.9790
    Epoch 5/10
    24/24 [==============================] - 104s 4s/step - loss: 0.2851 - accuracy: 0.9780 - val_loss: 0.1958 - val_accuracy: 0.9820
    Epoch 6/10
    24/24 [==============================] - 103s 4s/step - loss: 0.2695 - accuracy: 0.9783 - val_loss: 0.2905 - val_accuracy: 0.9790
    Epoch 7/10
    24/24 [==============================] - 103s 4s/step - loss: 0.3040 - accuracy: 0.9770 - val_loss: 0.3226 - val_accuracy: 0.9760
    Epoch 8/10
    24/24 [==============================] - 103s 4s/step - loss: 0.2572 - accuracy: 0.9810 - val_loss: 0.2654 - val_accuracy: 0.9820
    Epoch 9/10
    24/24 [==============================] - 104s 4s/step - loss: 0.1766 - accuracy: 0.9867 - val_loss: 0.2159 - val_accuracy: 0.9860
    Epoch 10/10
    24/24 [==============================] - 104s 4s/step - loss: 0.1946 - accuracy: 0.9840 - val_loss: 0.3092 - val_accuracy: 0.9740

``` python
# create a 2nd train generator which does not use data augmentation
# to get the true train accuracy
train_generator2 = gen_test.flow_from_directory(
  train_path,
  target_size=IMAGE_SIZE,
  batch_size=batch_size,
)
model.evaluate_generator(
    train_generator2,
    steps=int(np.ceil(len(image_files) / batch_size)))
```

    Found 3000 images belonging to 2 classes.

    [0.09657006577390703, 0.993]

``` python
# loss
plt.plot(r.history['loss'], label='train loss')
plt.plot(r.history['val_loss'], label='val loss')
plt.legend()
plt.show()
```

![](TF2_0_Transfer_Learning_with_Data_Augmentation_files/figure-gfm/cell-29-output-1.png)

``` python
# accuracies
plt.plot(r.history['accuracy'], label='train acc')
plt.plot(r.history['val_accuracy'], label='val acc')
plt.legend()
plt.show()
```

![](TF2_0_Transfer_Learning_with_Data_Augmentation_files/figure-gfm/cell-30-output-1.png)

<center>

<a href=''> ![Logo](../logo1.png) </a>

</center>
<center>
<em>Copyright Qalmaqihir</em>
</center>
<center>
<em>For more information, visit us at
<a href='http://www.github.com/qalmaqihir/'>www.github.com/qalmaqihir/</a></em>
</center>
