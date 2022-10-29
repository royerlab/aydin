Search.setIndex({docnames:["api/feature_generator","api/image_translator","api/image_translators/base","api/image_translators/classic","api/image_translators/cnn","api/image_translators/fgr","api/introduction","api/io","api/io/datasets","api/io/folders","api/io/io","api/nn","api/nn/jinet","api/nn/unet","api/options_json","api/regressors","api/regressors/cb","api/regressors/lgbm","api/regressors/linear","api/regressors/perceptron","api/regressors/random_forest","api/regressors/support_vector","api/restoration","api/restorations/deconvolve_restoration","api/restorations/denoise_restoration","api/transforms","contact_us/github","contact_us/imagesc","getting_started/hardware_requirements","getting_started/install","getting_started/why_aydin","index","tutorials/api_tutorials","tutorials/api_tutorials/image_translator/classic","tutorials/api_tutorials/image_translator/cnn","tutorials/api_tutorials/image_translator/fgr","tutorials/api_tutorials/image_translator/noise2noise","tutorials/api_tutorials/image_translator/supervised","tutorials/api_tutorials/restoration/classic","tutorials/api_tutorials/restoration/lucyrichardson","tutorials/api_tutorials/restoration/noise2self_cnn","tutorials/api_tutorials/restoration/noise2self_fgr","tutorials/api_tutorials/transforms/attenuation","tutorials/api_tutorials/transforms/deskew","tutorials/api_tutorials/transforms/fixed_pattern","tutorials/api_tutorials/transforms/highpass","tutorials/api_tutorials/transforms/histogram","tutorials/api_tutorials/transforms/motion","tutorials/api_tutorials/transforms/padding","tutorials/api_tutorials/transforms/periodic","tutorials/api_tutorials/transforms/range","tutorials/api_tutorials/transforms/salt_pepper","tutorials/api_tutorials/transforms/variance_stabilisation","tutorials/cli_tutorials","tutorials/gui_tutorials","tutorials/tutorials_home","use_cases/basics","use_cases/confocal","use_cases/introduction"],envversion:{"sphinx.domains.c":2,"sphinx.domains.changeset":1,"sphinx.domains.citation":1,"sphinx.domains.cpp":4,"sphinx.domains.index":1,"sphinx.domains.javascript":2,"sphinx.domains.math":2,"sphinx.domains.python":3,"sphinx.domains.rst":2,"sphinx.domains.std":2,"sphinx.ext.intersphinx":1,"sphinx.ext.viewcode":1,sphinx:56},filenames:["api/feature_generator.rst","api/image_translator.rst","api/image_translators/base.rst","api/image_translators/classic.rst","api/image_translators/cnn.rst","api/image_translators/fgr.rst","api/introduction.rst","api/io.rst","api/io/datasets.rst","api/io/folders.rst","api/io/io.rst","api/nn.rst","api/nn/jinet.rst","api/nn/unet.rst","api/options_json.rst","api/regressors.rst","api/regressors/cb.rst","api/regressors/lgbm.rst","api/regressors/linear.rst","api/regressors/perceptron.rst","api/regressors/random_forest.rst","api/regressors/support_vector.rst","api/restoration.rst","api/restorations/deconvolve_restoration.rst","api/restorations/denoise_restoration.rst","api/transforms.rst","contact_us/github.rst","contact_us/imagesc.rst","getting_started/hardware_requirements.rst","getting_started/install.rst","getting_started/why_aydin.rst","index.rst","tutorials/api_tutorials.rst","tutorials/api_tutorials/image_translator/classic.rst","tutorials/api_tutorials/image_translator/cnn.rst","tutorials/api_tutorials/image_translator/fgr.rst","tutorials/api_tutorials/image_translator/noise2noise.rst","tutorials/api_tutorials/image_translator/supervised.rst","tutorials/api_tutorials/restoration/classic.rst","tutorials/api_tutorials/restoration/lucyrichardson.rst","tutorials/api_tutorials/restoration/noise2self_cnn.rst","tutorials/api_tutorials/restoration/noise2self_fgr.rst","tutorials/api_tutorials/transforms/attenuation.rst","tutorials/api_tutorials/transforms/deskew.rst","tutorials/api_tutorials/transforms/fixed_pattern.rst","tutorials/api_tutorials/transforms/highpass.rst","tutorials/api_tutorials/transforms/histogram.rst","tutorials/api_tutorials/transforms/motion.rst","tutorials/api_tutorials/transforms/padding.rst","tutorials/api_tutorials/transforms/periodic.rst","tutorials/api_tutorials/transforms/range.rst","tutorials/api_tutorials/transforms/salt_pepper.rst","tutorials/api_tutorials/transforms/variance_stabilisation.rst","tutorials/cli_tutorials.rst","tutorials/gui_tutorials.rst","tutorials/tutorials_home.rst","use_cases/basics.rst","use_cases/confocal.rst","use_cases/introduction.rst"],objects:{"aydin.features.base":{FeatureGeneratorBase:[0,0,1,""]},"aydin.features.base.FeatureGeneratorBase":{compute:[0,1,1,""],create_feature_array:[0,1,1,""],get_receptive_field_radius:[0,1,1,""],load:[0,1,1,""],save:[0,1,1,""]},"aydin.features.extensible_features":{ExtensibleFeatureGenerator:[0,0,1,""]},"aydin.features.extensible_features.ExtensibleFeatureGenerator":{add_feature_group:[0,1,1,""],clear_features:[0,1,1,""],compute:[0,1,1,""],create_feature_array:[0,1,1,""],get_num_features:[0,1,1,""],get_receptive_field_radius:[0,1,1,""],load:[0,1,1,""],save:[0,1,1,""]},"aydin.it.transforms.attenuation":{AttenuationTransform:[25,0,1,""]},"aydin.it.transforms.deskew":{DeskewTransform:[25,0,1,""]},"aydin.it.transforms.fixedpattern":{FixedPatternTransform:[25,0,1,""]},"aydin.it.transforms.highpass":{HighpassTransform:[25,0,1,""]},"aydin.it.transforms.histogram":{HistogramEqualisationTransform:[25,0,1,""]},"aydin.it.transforms.motion":{MotionStabilisationTransform:[25,0,1,""]},"aydin.it.transforms.padding":{PaddingTransform:[25,0,1,""]},"aydin.it.transforms.periodic":{PeriodicNoiseSuppressionTransform:[25,0,1,""]},"aydin.it.transforms.salt_pepper":{SaltPepperTransform:[25,0,1,""]},"aydin.it.transforms.variance_stabilisation":{VarianceStabilisationTransform:[25,0,1,""]},"aydin.regression.linear":{LinearRegressor:[18,0,1,""]},"aydin.regression.linear.LinearRegressor":{fit:[18,1,1,""],load:[18,1,1,""],predict:[18,1,1,""],recommended_max_num_datapoints:[18,1,1,""],save:[18,1,1,""],stop_fit:[18,1,1,""]},"aydin.regression.random_forest":{RandomForestRegressor:[20,0,1,""]},"aydin.regression.random_forest.RandomForestRegressor":{fit:[20,1,1,""],load:[20,1,1,""],predict:[20,1,1,""],recommended_max_num_datapoints:[20,1,1,""],save:[20,1,1,""],stop_fit:[20,1,1,""]},"aydin.regression.support_vector":{SupportVectorRegressor:[21,0,1,""]},"aydin.regression.support_vector.SupportVectorRegressor":{fit:[21,1,1,""],load:[21,1,1,""],predict:[21,1,1,""],recommended_max_num_datapoints:[21,1,1,""],save:[21,1,1,""],stop_fit:[21,1,1,""]}},objnames:{"0":["py","class","Python class"],"1":["py","method","Python method"]},objtypes:{"0":"py:class","1":"py:method"},terms:{"0":[0,14,18,20,25,42,43,47],"0001":[18,20],"01":[14,25],"08":25,"09":25,"1":[14,18,25,43,44],"100":20,"1024":20,"11":25,"12":25,"125":25,"13":14,"16":28,"16gb":31,"17":48,"2":[14,25,44],"20":39,"2048":[14,20],"24":53,"2d":25,"3":[14,25,29,31,43],"32":[14,20,25,28],"321":[25,53],"3d":25,"4":[25,28,31,53],"45":25,"481":53,"5":[14,25],"512":[14,18,20],"64":25,"8":25,"9":[25,29,31],"95":14,"abstract":[0,1],"boolean":0,"case":[25,30,31,54],"class":[0,1,14,18,20,21,25,38,40,41],"default":[25,33,34,35,38,39,40,41,53,54],"do":[25,26,33,34,35,38,40,41,58],"final":[29,54],"float":[0,18,20,25],"function":[0,6,14,18,20,21,25],"import":[33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52],"int":[0,18,20,21,25],"new":[14,26,54,58],"null":14,"public":0,"return":[0,18,20,21],"static":[0,18,20,21],"true":[0,14,21,25,53,58],"try":32,"while":[25,30,31,54,58],And:25,As:54,At:54,By:[1,54],For:[25,54],If:[0,18,20,21,25,26,27,29,31,54],In:[25,31,54],It:[25,29,30,31,33,34,35,36,37,38,40,41,54],On:29,One:[22,33,34,35,38,39,40,41,53],Or:53,TO:53,The:[18,20,21,25,29,58],Then:[29,54],There:[54,58],To:[14,25,31,54],With:[29,54],abl:[18,20,21,28,29,54],about:[36,53,54,58],absenc:31,access:[53,54],acheiv:0,achiev:[0,25,37,53],acquir:54,across:54,activ:[29,58],ad:[0,25,54],add:[0,25,54],add_feature_group:0,add_transform:[33,34,35,36,37],addit:0,address:54,adjust:[54,58],advanc:25,advantag:[18,20,25],after:[25,29,33,34,35,38,40,41,54],again:54,aggress:25,al:29,algorithm:[0,25,28,31,54,58],align:25,all:[0,18,20,21,25,30,31,55,58],allevi:25,alloc:0,allow:25,along:25,alpha:18,also:[0,18,25,26,31,32,33,34,35,38,39,40,41,53,54],altern:53,alwai:[54,58],among:58,amount:25,an:[0,18,20,21,25,28,29,31,32,36,53,54,55,58],ani:[0,1,25,26,27,30,54,58],anoth:[18,20,21],anscomb:[25,52],anywher:54,api:[0,1,22,23,24,31,33,34,35,36,37,42,43,44,45,46,47,48,49,50,51,52,55],app:29,appli:[0,18,20,21,25,30,32,54],applic:[29,55],approach:[0,25,29,53],approxim:25,ar:[0,1,25,28,29,54,58],arbitrari:31,arg:[0,53],argument:[0,54],around:25,arrai:[0,25,31,58],arraylik:[18,20,21],artifact:[25,54],asid:28,assum:[25,53],attenu:[25,32],attenuation_transform:42,attenuationtransform:42,attract:20,auto:31,automat:25,avail:[31,36,55],averag:25,awai:25,ax:[25,42,44,47,53],axi:[25,53],aydin:[0,1,6,7,14,18,20,21,22,25,28,31,32,33,34,35,36,37,38,40,41,42,43,44,45,46,47,48,49,50,51,52,55],aydin_env:29,back:54,backend:39,background:[14,25],balance_training_data:14,base:[0,1,25],basic:[14,54,58],batch:[31,53],batch_ax:53,batteri:[0,18,20,21],becaus:25,befor:[25,30,33,34,35,38,40,41,54],being:54,believ:54,below:[1,14,29,32,36,37,53,54,58],benefici:25,best:[25,26,27,31],beta:18,better:[21,31],between:[20,25],beyond:58,bigger:28,blank:0,blind:[0,25],blink:25,blue:25,blur:25,blurri:25,bool:[0,21,25],boost:21,boot:54,border:25,both:[25,31,53,54,58],box:[25,28],brew:29,bright:25,bring:54,broken:25,browser:30,bug:31,built:[6,54],bullet:58,bundl:54,butterworth:28,button:[29,54],call:[18,20,21,22,25,54],callback:[18,20,21],came:58,camera:25,can:[0,1,11,14,18,20,21,22,25,26,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,54,58],canvas:54,card:[28,31],care:25,carefulli:[25,30],carri:25,catboost:[15,28],caus:[18,20,21,25,28],cb:14,cbregressor:14,center:0,central:0,certain:[25,29,54],challeng:[25,58],chang:54,channel:[0,18,20,21,31],channel_ax:53,characteris:25,characterist:54,checbox:54,check:[31,58],checkbox:54,chemistri:58,chill:31,choic:[53,54],choos:[29,58],chosen:[14,25,53,54],clahe:25,classic:[32,33],classic_denois:38,classic_restor:38,clear:0,clear_featur:0,cli:[14,31,54,55],click:[29,54],close:[25,29],closest:25,cnn:[34,37],code:[31,36],colab:32,colors_of_nois:25,column:[25,54],com:20,combin:[25,30],come:[31,54,58],command:[29,30,31,53,55],common:[25,54],commun:[26,31],comparison:30,complementari:1,complet:25,complex:25,complic:25,compress:[25,29],comput:[0,6,25,28,29,54],compute_load:14,conda:28,configur:[6,30,54],confoc:58,consid:[0,25,31,54],consist:25,constant:25,contact:58,continu:29,contrast:58,contribut:26,control:25,conveni:29,convent:54,cool:54,cooler:54,coordin:0,copi:47,core:[28,31],correct:[25,51,54],correctli:54,correl:25,correspond:[18,20,21,54],could:28,coupl:11,cox:25,cpu:[28,31],craft:30,creat:[0,26,29,54],create_feature_arrai:0,crop:25,cuda:28,cudatoolkit:[28,29],cumul:54,cupi:[39,53],current:[1,25,30,31,54],custom:[30,31],czi:54,d:0,dark:25,datapoint:[18,20,21],dataset:[7,18,20,21,54,58],dct_max_freq:14,decent:20,decid:54,decimate_large_scale_featur:14,deconvolut:[22,53],deconvolv:[1,22,32],decreas:25,defin:25,degrad:58,delta:[25,43],demonstr:[32,53],denois:[0,1,22,25,28,31,32,36,39],denoised_imag:[38,40,41],densiti:25,depend:[25,28,29],depth:55,deriv:25,descript:55,design:[54,58],desir:[53,54],deskew:[25,32],deskew_transform:43,deskewed_imag:43,deskewtransform:43,destin:29,detail:[11,29,54],detect:54,detector:[25,58],develop:0,dialog:54,dict:14,differ:[0,11,25,29,31,36,53,54,58],difficulti:58,dim:25,dimens:[0,25,31,53],dimension:[31,53,54,58],directli:0,discuss:26,disentangl:25,disk:58,dive:54,divers:58,doe:[21,25],done:[29,54],doubl:[29,54],download:[29,31],drag:54,dramat:30,drop:54,dtype:[14,53],dump:14,durat:54,dure:25,e:[0,25],each:[25,30,32,54,58],easi:[20,30,33,34,35,36,37,38,40,41,54],easili:[14,31],edg:25,effect:[25,30,54],either:25,element:54,en:25,enabl:0,encod:[14,18,20,21],end:[28,54],end_valu:25,endpoint:[38,39,40,41],engin:0,entir:0,entri:54,equal:25,equalis:[25,32],especi:[25,31],essenti:58,estim:25,etc:58,even:31,exact:54,exampl:[14,25,30,36,37,53],except:54,exclud:0,exclude_center_featur:0,exclude_center_valu:0,excluded_voxel:0,execut:29,exhaust:31,exist:28,expect:53,experi:[21,31],explain:54,explan:[30,54],explanatori:54,expos:54,express:0,ext:53,extend:[0,1,58],extend_large_scale_featur:14,extens:[0,54],extensible_featur:0,extent:25,extract:29,extrem:25,f4:14,f:[18,20,21],face:58,facilit:25,fairli:54,fals:[0,14,25,53],familiar:29,far:25,fast:[18,20,25,28,31],faster:[28,31],favour_bright_pixel:14,feasibl:54,featur:[6,14,18,20,21,31,32,53,54,55],feature_gener:14,feature_group:0,feature_last_dim:0,featuregroupbas:0,featuris:0,feel:[26,54],few:[25,53],fgr:[14,35,36,37],field:0,figur:30,file:[14,29],fill:58,filter:25,find:[1,29,30,31,36,54],fine:54,first:[0,26,29,53,54],fit:[18,20,21],fix:[25,32],fix_lipschitz:25,fix_rep:25,fixedpattern:[25,44],fixedpattern_transform:44,fixedpatterntransform:44,flexibl:29,fluctuat:25,folder:[0,7,18,20,21,29],follow:[6,11,31,33,34,35,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53],forest:[15,21],forget:26,form:25,format:[14,53,54],forrest:20,forum:27,forward:54,found:[11,31,36],four:53,frame:25,free:26,frequenc:25,friendli:[31,54,55],from:[0,6,18,20,21,25,28,29,31,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,54,58],frozen:[0,18,20,21],full:54,fundament:[54,58],further:[25,58],futur:54,g:25,gaussian:25,gb:28,gener:[1,6,18,25,58],get:[29,53],get_num_featur:0,get_receptive_field_radiu:0,gigabyt:31,github:[20,26,31,36],give:[25,28,54,55],given:[0,18,20,21,25,30,31,54],glob:53,go:31,good:25,googl:32,gpu:[14,28,29,31],gpu_devic:14,gradient:21,graphic:[28,31,55],great:36,groundtruth:37,group:0,gui:[14,53,54,55],guid:29,guidelin:26,h:0,ha:[20,25,29],handi:54,handl:[31,54],happen:25,have:[11,14,18,20,21,25,28,29,31,32,53,54,58],header:54,headless:55,help:[14,25,28,29,53,54],henc:[30,54],here:[30,31,36,53,58],high:[25,28,32],highli:[25,30],highlight:[28,54],highpass:[25,45],highpass_transform:45,highpasstransform:45,histogram:[25,32],histogram_transform:46,histogramequalisationtransform:46,hold:58,home:29,how:[32,53,54,58],howev:[18,54],href:[20,25],http:[20,25],huber:18,i:0,icon:54,idea:25,ideal:[25,28,58],imag:[0,6,14,22,25,27,28,31,36,42,43,44,45,46,47,48,49,50,51,52],image1:53,image2:53,image3:53,image_model:53,image_opt:53,imagedenoiserclass:[1,32],imagetranslatorbas:1,imagetranslatorcnn:[1,32,37],imagetranslatorfgr:[1,14,32,37],impact:25,imperfect:54,implement:[0,1,11,20,22,30,32,54],importantli:25,improv:[18,20,21,25,30,54,58],impuls:25,includ:[0,18,20,21,54],include_corner_featur:14,include_dct_featur:14,include_fine_featur:14,include_line_featur:14,include_median_featur:14,include_random_conv_featur:14,include_scale_on:14,include_spatial_featur:14,increas:25,independ:54,individu:30,inevit:54,infer:[18,20,21,54],info:53,inform:[11,36,53],inher:58,inlcud:0,input:[0,18,20,21,54],inspect:14,instal:[28,31,54],instanc:36,instanci:[18,20,21],instead:25,instrument:58,integ:25,integr:[25,30,31,53,55],intend:54,intens:25,interact:29,interest:58,interfac:[31,55],intern:[0,18,20,21,30],interpol:25,interpret:54,io:6,is_fold:53,issu:[25,26,31,54,58],it_transform:[38,40,41],item:28,iter:25,its:[25,29,54],itself:25,jinetmodel:11,job:31,johnson:25,json:[18,20,21,53,54],jsonpickl:14,just:[29,54,58],keep:25,keyword:0,kind:[25,58],kindli:31,know:[54,58],kwarg:[0,14,18,25,38,40,41],l1:[14,20],laptop:31,larg:[25,28,31],larger:25,last:[0,54],layer:6,lazi:0,lead:25,learn:[18,20,21,58],learning_r:[14,20],least:[25,28,31,54],leave_as_float:25,left:[0,25,54],less:25,lesser:25,level:[6,25,53],libomp:29,librai:20,librari:[28,54],lightgbm:[15,20],lightweight:30,like:[26,27,54,58],line:[29,31,33,34,35,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,55],linear:[15,21,25],linear_ramp:25,linearregressor:18,link:30,lipschitz:25,list:[0,25,54,58],littl:25,load:[0,18,20,21,53],local:25,locat:54,log:25,look:54,loss:[14,20],lossi:25,lossless:53,lot:58,low:25,lower:53,lowpass:33,lr:39,lr_deconvolved_imag:39,luci:30,lucyrichardson:[39,53],machin:[28,31,53],made:[1,53],mai:[25,28],main:55,make:[28,29,54],manag:29,mani:[14,25,58],map:0,mask_radiu:25,materi:58,max:25,max_bin:[14,20],max_level:14,max_num_estim:[14,20],max_num_iter:[18,39],max_num_peak:25,max_pixel_shift:25,max_rep:25,max_voxels_for_train:14,maximum:[18,20,21,25],mean:25,measur:58,mechan:58,median:25,median_filt:25,memori:[0,28],mention:54,menubar:54,metadata:[53,54],method:[1,22,30,33,36,58],microscopi:58,microsoft:20,might:[0,25,28,54],mileston:26,millisecond:53,min_const:25,min_length_to_pad:25,min_level:14,min_num_estim:14,minimum:25,minmax:50,mistak:54,modal:[25,54,58],mode:[18,25,50,52],model:[11,36],models_to_us:[18,20,21],modest:[28,31],modul:[1,7,22],more:[14,25,28,29,31,36,54,58],most:[1,25,54],motion:[25,32],motion_transform:47,motionstabilisationtransform:47,multi:[6,54,58],multipl:[1,18,20,21,25,54,58],multipli:0,must:25,n2:[0,25,40,41],n:[0,29,31,54],name:29,napari:[30,53,54],nb_featur:0,nd:30,ndarrai:0,ndim:0,need:[25,28,31,54],neighbor:25,network:11,neural:11,never:0,next:[25,54],nn:[6,53],nois:[25,32,36,58],noise2nois:32,noise2self_cnn:40,noise2self_fgr:41,noise2selfcnn:32,noise2selffgr:[14,32,53],noisi:[33,34,35,36,37],noisy_and_blurred_imag:39,noisy_imag:[36,38,40,41],noisy_image1:36,noisy_image2:36,noisy_pair:36,non:25,none:[0,18,20,21,25],nor:58,note:[25,53,58],noth:25,notion:1,now:25,num_iter:25,num_leav:[14,20],num_reserved_featur:0,num_sinusoidal_featur:14,numba:28,number:[0,18,20,21,31,54],numer:54,numpi:[0,14,18,20,21],nvidia:[28,31],object:[18,20,21,22,23,24,33,34,35,38,39,40,41],observ:54,obvious:31,off:20,offend:25,offer:[1,20],offlin:31,offset:[0,25],often:25,oftentim:54,older:28,onc:[29,54],one:[1,25,53,54],onli:[25,30,31,53,54],open:[29,31,54],oper:[25,29,55],optic:58,option:[0,25,33,34,35,38,39,40,41],orang:28,order:25,org:25,orient:[22,23,24,33,34,35,38,39,40,41],origin:[25,36],other:[6,25,28,31,58],ouput:[18,20,21],our:[26,31,33,34,35,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,54,58],out:[1,28,54],outlier:25,over:[25,54],overview:28,packag:[6,14,28,29,30,31],pad:[25,32],pad_mod:25,pad_width:[25,48],padding_transform:48,paddingtransform:[33,34,35,36,37,38,40,41,48],page:[11,28,29,58],pair:[18,20,21,36],paper:36,paramet:[0,18,20,21,25,30,31,53,54,55,58],part:[1,25,54],particip:26,partit:29,pass:[0,14,25,32,33,34,35,38,40,41,53,54],passthrough_channel:0,patch:0,path:[0,18,20,21,53],patienc:[14,20,31],pattern:[25,32,53],peak:25,pepper:[25,32],per:25,percentil:25,perceptron:15,perform:[1,18,20,21,25,28,31,54],period:[25,32],periodic_noise_suppression_transform:49,periodicnoisesuppressiontransform:49,phenomena:25,photo:58,pip:[29,54],pixel:[0,25],place:[22,54],placement:25,plai:53,plan:26,plane:25,platform:32,pleas:[25,29,31,54,58],png:54,point:[25,54,58],poisson:25,poorli:18,popul:58,possibl:[0,25,53,54],post:[25,58],post_processing_is_invers:25,postprocess:[42,43,45,46,47,48,49,50,52,54],postprocessed_imag:47,pour:21,power:25,pre:58,predict:[18,20,21],prefer:28,prepar:[29,36],preprocess:[25,42,43,44,45,46,47,48,49,50,51,52,54],preprocessed_imag:47,present:58,pretti:[30,54],previou:54,prioriti:25,problem:[26,58],problemat:28,procedur:[22,23,24,38,39,40,41],process:[25,28,58],profil:31,program:[31,55],progress:29,proport:25,provid:[0,1,6,22,25,29,30,31,32,36,37,53,54,55],psf:[30,53],psf_kernel:39,publish:36,pull:26,purpos:6,py:14,python:[14,29,31,55],qualiti:[20,30],quick:37,quit:[28,36,37],radiu:0,ram:[28,31],ramp:25,random:[15,21],random_forest:20,randomforestregressor:20,rang:[25,32],range_transform:50,rangetransform:[33,34,35,36,37,38,40,41,50],rapidli:54,reach:[26,27],read:[26,54],readi:29,realis:58,reappli:25,reason:25,recent:28,recept:0,recommend:[18,20,21,25,28,30,31],recommended_max_num_datapoint:[18,20,21],recov:25,red:28,reduc:[14,25],refer:[25,54],reference_index:25,reflect:25,region:25,regist:25,registr:25,regress:[14,18,20,21],regressor:[6,14],regressor_callback:[18,20,21],rel:[0,1,25],relat:[26,53,54,58],releas:31,reliabl:58,remov:25,replac:25,report:31,repositori:[26,31],repressor:18,request:31,requir:[1,29,58],resist:58,respect:54,rest:54,restor:[6,39],result:[0,25,54],revert:54,rich:31,richardson:30,right:[0,29,54],row:[25,54],rtx:28,run:[14,28,29,30,31,37,53,54],runtim:28,s:[0,28],said:28,salt:[25,32],salt_pepp:[25,51],salt_pepper_transform:51,saltpeppertransform:[25,51],same:[0,14,36,37,54],save:[0,14,18,20,21,53],sc:27,scale:[0,25],scale_one_width:14,scipi:39,screen:[53,54],screenshot:54,script:[29,55],second:[25,29],section:54,see:[29,36,37,54],seen:25,select:[30,53],self:[25,31,54],separ:[0,54],sequenc:25,seri:28,set:[0,25,29,53,58],setup:54,sever:32,sh:29,shape:[0,53,54],share:[14,36],shear:25,shift:25,shorten:54,should:[25,29,53],show:[29,54],shown:[29,37,53,54],sigma:25,signal:[25,36],significantli:54,silver:58,similar:37,simpl:[18,25,29],simplest:18,simpli:25,simplif:25,singl:[33,34,35,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,54,58],singular:58,situat:18,size:[0,31],skew:25,skew_axi:[25,43],skewed_imag:43,slice:54,slow:[21,28],small:54,smaller:25,snap:25,snippet:30,so:[22,29,54],solv:25,some:[18,25,28,30,31],sometim:25,sourc:[0,18,20,21,25],space:25,spatial:0,spatial_feature_offset:0,spatial_feature_scal:0,spatial_features_coarsen:14,spatio:31,specif:[28,29,33,34,35,38,40,41,53],specifi:[0,25,54],spectral:25,speed:20,spin:58,spot:[0,25],sqrt:25,ss:30,stabil:25,stabilis:[25,32],stack:[25,54],standard_featur:14,standardfeaturegener:14,start:[25,29,30,58],step:[29,58],still:25,stop:[18,20,21],stop_fit:[18,20,21],store:[0,14],str:[0,18,20,21,25],strength:28,strong:25,structur:31,studio:[14,28,31,53,55],subtab:54,success:29,suffici:28,summari:[1,28],supervis:[25,31,32],support:[15,25,28,30,31,54],support_vector:21,supportvectorregressor:21,suppress:[25,32],sure:[29,54],symmetr:25,system:29,t:[25,54],tab:53,tabl:30,tackl:25,take:0,target:[18,20,21],task:[1,11,25],tbf:8,teh:0,tell:[53,54],tempor:31,termin:[29,54],test:[18,20,21,30],than:[21,25],thaw:[0,18,20,21],thei:[14,53,54],them:[32,54],thi:[0,1,18,20,21,25,29,30,53,54,58],those:54,thread:[18,20,21],three:55,threshold:25,through:[0,31],thu:25,tif:[53,54],time:[25,30,54],timelaps:25,tini:31,todo:30,togeth:54,toler:25,too:[21,25,28],tool:31,toolkit:28,tooltip:54,top:[6,54],tour:31,track:26,trade:20,train:[18,20,21,33,34,35,36,37,38,39,40,41],transform:[6,30,33,34,35,38,40,41,54],translat:[0,6,33,34,35,36],treat:[25,53],tri:[28,54],tricki:25,trigger:14,troubl:28,tune:[25,31,54],tupl:[0,14],turn:25,tutori:31,two:[25,29,53,54],type:[14,18,20,21],typic:[25,31],tyx:25,uint8:53,unadvis:25,unbalanc:25,understand:[14,31,54,58],undon:25,unetmodel:11,union:25,unit:28,unless:25,unsupervis:31,until:25,untouch:25,unzip:29,up:[31,54],updat:58,upstream:6,us:[0,1,6,11,20,22,25,26,27,28,29,31,32,42,43,44,45,46,47,48,49,50,51,52,53,54],useful:25,user:[29,30,31,53,54,55],valid:[18,20,21],valu:[18,20,21,25,54],vari:[6,25],varianc:[25,32],variance_stabilis:[25,52],variancestabilisationtransform:52,variant:[14,54],variat:25,variou:[11,22,32,53],ve:54,vector:[0,15,25],verbos:20,veri:[18,20,25,28,31],versatil:30,via:[29,54],volum:25,voxel:25,voxel_keep_ratio:14,vst:[25,52],w:0,wai:[0,25,26,27,29],want:[0,25,54],we:[1,6,11,14,25,29,30,31,32,53,54,58],weak:28,welcom:[29,54],well:[0,25,30,54,58],what:58,when:[0,14,18,20,21,25,54],whenev:28,where:[0,31,53,54],wherea:28,which:[0,1,6,25,29,54],who:0,wiki:25,wikipedia:25,window:[30,54],without:54,work:[25,28,31,53],wors:25,worth:54,would:[53,54],write:54,x:[18,20,21,54],x_test:[18,20,21],x_train:[18,20,21],x_valid:[18,20,21],xy:25,y:[18,20,21,29,54],y_test:[18,20,21],y_train:[18,20,21],y_valid:[18,20,21],yeo:25,you:[1,14,25,26,27,29,30,31,32,36,37,42,43,44,45,46,47,48,49,50,51,52,53,54,58],your:[28,29,31,54],yx:53,z:[25,54],z_axi:[25,43],zarr:54,zero:29,zip:53},titles:["Feature Generator","Image Translator","ImageTranslatorBase","ImageDenoiserClassic - Classic Image Denoiser","ImageTranslatorCNN - Convolutional Neural Network","ImageTranslatorFGR - Feature Generation&amp;Regression","API Introduction","IO","datasets","folders","IO","NN","JINetModel","UNetModel","Options JSON","Regressors","CatBoost Regressor","LightGBM Regressor","Linear Regressor","Perceptron Regressor","Random Forest Regressor","Support Vector Regressor","Restoration","Deconvolve Restoration","Denoise Restoration","Transforms","Report a bug/Request a feature","Ask general questions about Aydin","Hardware Requirements","Install Aydin","Why Aydin?","Getting started, with bundles:","API Tutorials","Denoising an image using ImageDenoiserClassic","Denoising an image using ImageTranslatorCNN","Denoising an image using ImageTranslatorFGR","How to implement Noise2Noise using ImageTranslatorFGR","How to implement supervised denoising using image translators","Denoising an image using Classic restoration API","Deconvolving an image using aydin API","Denoising an image using Noise2SelfCNN restoration API","Denoising an image using Noise2SelfFGR restoration API","Apply Attenuation transform","Apply Deskew transform","Apply fixed pattern transform","Apply high pass transform","Apply histogram equalisation transform","Apply motion stabilisation transform","Apply padding transform","Apply periodic noise suppression transform","Apply range transform","Apply salt-pepper transform","Apply variance stabilisation transform","Aydin CLI Tutorials","Aydin Studio Tutorials","Tutorials Home","Denoising Basics with Aydin.","Denoising Spinning-Disk Confocal Microscopy Images with Aydin","Aydin Use-Cases, and the Art &amp; Science of Image Denoising."],titleterms:{"case":58,about:27,access:30,all:54,an:[33,34,35,38,39,40,41],api:[6,30,32,38,39,40,41],appli:[42,43,44,45,46,47,48,49,50,51,52],art:58,ask:27,attenu:42,attenuationtransform:25,automat:30,aydin:[27,29,30,39,53,54,56,57,58],backend:53,basic:56,blind:30,bug:26,bundl:[29,31],can:53,catboost:16,channel:54,check:53,choos:[53,54],cite:31,classic:[3,24,38],cli:[30,53],comput:53,conda:29,confoc:57,convolut:4,crop:54,cuda:29,custom:53,data:30,dataset:8,deconvolut:30,deconvolv:[23,39,53],denois:[3,24,30,33,34,35,37,38,40,41,53,54,56,57,58],deskew:43,deskewtransform:25,detect:30,dimens:54,disk:57,document:31,environ:29,equalis:46,extensiblefeaturegener:0,featur:[0,5,26],featuregeneratorbas:0,file:[53,54],fix:44,fixedpatterntransform:25,folder:[9,53,54],forest:20,gener:[0,5,27],get:31,gui:30,halucin:30,hardwar:28,helper:30,high:45,highpasstransform:25,histogram:46,histogramequalisationtransform:25,home:55,how:[31,36,37],hyperstack:[30,54],imag:[1,3,30,32,33,34,35,37,38,39,40,41,53,54,57,58],imagedenoiserclass:[3,33],imagetranslatorbas:2,imagetranslatorcnn:[4,34],imagetranslatorfgr:[5,35,36],implement:[36,37],instal:29,interpret:53,introduct:6,io:[7,10],jinetmodel:12,json:14,less:30,let:54,lightgbm:17,linear:18,linux:29,load:54,luci:23,mac:29,microscopi:57,model:[53,54],more:30,motion:47,motionstabilisationtransform:25,multipl:53,network:4,neural:4,nn:11,nois:49,noise2nois:36,noise2selfcnn:[24,40],noise2selffgr:[24,41],option:[14,53,54],osx:29,output:[53,54],own:53,pad:48,paddingtransform:25,pass:45,pattern:44,pepper:51,perceptron:19,period:49,periodicnoisesuppressiontransform:25,pip:31,post:54,postprocess:30,pre:[53,54],preprocess:30,preview:54,process:54,question:27,random:20,rang:50,rangetransform:25,read:53,regress:5,regressor:[15,16,17,18,19,20,21],report:26,request:26,requir:[28,31],restor:[22,23,24,32,38,40,41],richardson:23,right:53,robust:30,s:54,salt:51,saltandpeppertransform:25,save:54,scienc:58,select:54,set:54,singl:53,specif:54,specifi:53,spin:57,split:54,spot:30,stabilis:[47,52],standardfeaturegener:0,start:[31,54],studio:54,supervis:37,support:[21,29],suppress:49,tab:54,train:[53,54],transform:[25,32,42,43,44,45,46,47,48,49,50,51,52],translat:[1,32,37],tutori:[32,53,54,55],unetmodel:13,us:[30,33,34,35,36,37,38,39,40,41,58],varianc:52,variancestabilisationtransform:25,vector:21,via:30,view:[30,53],why:30,window:29,your:53}})