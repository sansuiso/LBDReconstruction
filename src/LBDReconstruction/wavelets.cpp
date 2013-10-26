//  wavelets.h
//
//	Copyright (C) 2011-2012  Signal Processing Laboratory 2 (LTS2), EPFL,
//	Emmanuel d'Angelo (emmanuel.dangelo@epfl.ch),
//  and Jérôme Gilles (jegilles@math.ucla.edu)
//
//  Redistribution and use in source and binary forms, with or without modification,
//  are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
//  This software is provided by the copyright holders and contributors "as is" and
//  any express or implied warranties, including, but not limited to, the implied
//  warranties of merchantability and fitness for a particular purpose are disclaimed.
//  In no event shall the Intel Corporation or contributors be liable for any direct,
//  indirect, incidental, special, exemplary, or consequential damages
//  (including, but not limited to, procurement of substitute goods or services;
//  loss of use, data, or profits; or business interruption) however caused
//  and on any theory of liability, whether in contract, strict liability,
//  or tort (including negligence or otherwise) arising in any way out of
//  the use of this software, even if advised of the possibility of such damage.

#include "wavelets.h"

#include <iostream>
#include <vector>
#include <algorithm>

static void 
Wavegeneral_h(cv::Mat& a, int JJ, int n, int isign, double *P, double *Q, int fsize)
{
  static float b[1024];
	
  if (n < fsize) return;
  int nh = n >> 1;
	
  if (isign >= 0) 
  {
    for (int i = 0,j = 0; j < n; j+=2, ++i) 
    {
      b[i]=0.0f; b[i+nh]=0.0f;
      for(int k = 0;k < fsize; ++k) 
      {
        b[i] += P[k] * a.at<float>(JJ, (j+k)%n);
        b[i+nh] += Q[k] * a.at<float>(JJ, (j+k)%n);
      }
    }
  } 
  else 
  {
    for (int i = 0, r = nh-(fsize/2-1); i < n; r+=1, i+=2) 
    {
      b[i]=0.0f;
      b[i+1]=0.0f;
			
      for(int k = fsize-2, j = 0; k >= 0; k-=2, j+=1) 
      {
        b[i] += P[k] * a.at<float>(JJ, ((r+j)%a.cols)%nh) + Q[k] * a.at<float>(JJ, nh+((r+j)%a.cols)%nh);
        b[i+1] += P[k+1] * a.at<float>(JJ, ((r+j)%a.cols)%nh) + Q[k+1] * a.at<float>(JJ, nh+((r+j)%a.cols)%nh);
      }
    }
  }
	
  for(int i = 0; i < n; ++i)
    a.at<float>(JJ, i) = b[i];
}

static void 
Wavegeneral_v(cv::Mat& a, int JJ, int n, int isign, double *P, double *Q, int fsize)
{
  static float b[1024];
	
  if (n < fsize) return;
  int nh = n >> 1;
	
  if (isign >= 0) 
  {
    for (int i = 0, j = 0; j < n; j+=2, ++i) 
    {
      b[i]=0.0f;
      b[i+nh]=0.0f;
			
      for(int k = 0; k < fsize; ++k) 
      {
        b[i] += P[k] * a.at<float>( (j+k)%n, JJ);
        b[i+nh] += Q[k] * a.at<float>( (j+k)%n, JJ);
      }
    }
  } 
  else 
  {
    for (int i = 0,r = nh-(fsize/2-1); i < n; r+=1, i+=2) 
    {
      b[i]=0.0f;
      b[i+1]=0.0f;
			
      for(int k = fsize-2, j = 0; k >= 0; k-=2, j+=1) 
      {
        b[i] += P[k] * a.at<float>(((r+j)%a.rows)%nh, JJ) + Q[k] * a.at<float>(nh+((r+j)%a.rows)%nh, JJ);
        b[i+1] += P[k+1] * a.at<float>(((r+j)%a.rows)%nh, JJ) + Q[k+1] * a.at<float>(nh+((r+j)%a.rows)%nh, JJ);
      }
    }
  }
  for(int i = 0; i < n; ++i)
    a.at<float>(i, JJ) = b[i];
}

void lts2::WaveletCoef(const std::string& name,double **P,double **Q,int *fsize) {
	
  double *p = NULL;
  double *q = NULL;
  int n = 0;
  int i;
	
  if (*P != 0)
    free(*P);
  if (*Q != 0)
    free(*Q);
	
  if (!strcmp(name.c_str(),"haar")) {
    n = 2;
    p = new double[n];
    q = new double[n];
		
    p[0] =   .70710678118654752440;
    p[1] =   .70710678118654752440;
		
    for(i=0;i<n;++i) {
      q[i] = p[n-1-i] * ((i-1)%2?-1:1);
    }
  } else if (!strcmp(name.c_str(),"daub4")) {
    n = 4;
    p = new double[n];
    q = new double[n];
		
    p[0] =   0.4829629131445341;
    p[1] =   0.8365163037378079;
    p[2] =   0.2241438680420134;
    p[3] = - 0.1294095225512604;
		
    for(i=0;i<n;++i) {
      q[i] = p[n-1-i] * ((i-1)%2?-1:1);
    }
  } else if (!strcmp(name.c_str(),"daub6")) {
    n = 6;
    p = new double[n];
    q = new double[n];
		
    p[0] =   0.33267055294999998;
    p[1] =   0.80689150931099995;
    p[2] =   0.45987750211799999;
    p[3] = - 0.13501102001000001;
    p[4] = - 0.085441273881999999;
    p[5] =   0.035226291882000001;
		
    for(i=0;i<n;++i) {
      q[i] = p[n-1-i] * ((i-1)%2?-1:1);
    }
  } else if (!strcmp(name.c_str(),"daub8")) {
    n = 8;
    p = new double[n];
    q = new double[n];
		
    p[0] =   0.230377813309;
    p[1] =   0.714846570553;
    p[2] =   0.630880767930;
    p[3] = - 0.027983769417;
    p[4] = - 0.187034811719;
    p[5] =   0.030841381836;
    p[6] =   0.032883011667;
    p[7] = - 0.010597401785;
		
    for(i=0;i<n;++i) {
      q[i] = p[n-1-i] * ((i-1)%2?-1:1);
    }
  } else if (!strcmp(name.c_str(),"daub10")) {
    n = 10;
    p = new double[n];
    q = new double[n];
		
    p[0] =   0.160102397974;
    p[1] =   0.603829269797;
    p[2] =   0.724308528438;
    p[3] =   0.138428145901;
    p[4] = - 0.242294887066;
    p[5] = - 0.032244869585;
    p[6] =   0.077571493840;
    p[7] = - 0.006241490213;
    p[8] = - 0.012580751999;
    p[9] =   0.003335725285;
		
    for(i=0;i<n;++i) {
      q[i] = p[n-1-i] * ((i-1)%2?-1:1);
    }
  } else if (!strcmp(name.c_str(),"daub12")) {
    n = 12;
    p = new double[n];
    q = new double[n];
		
    p[0] =   0.111540743350;
    p[1] =   0.494623890398;
    p[2] =   0.751133908021;
    p[3] =   0.315250351709;
    p[4] = - 0.226264693965;
    p[5] = - 0.129766867567;
    p[6] =   0.097501605587;
    p[7] =   0.027522865530;
    p[8] = - 0.031582039317;
    p[9] =   0.000553842201;
    p[10] =   0.004777257511;
    p[11] = - 0.001077301085;
		
    for(i=0;i<n;++i) {
      q[i] = p[n-1-i] * ((i-1)%2?-1:1);
    }
  } else if (!strcmp(name.c_str(),"daub14")) {
    n = 14;
    p = new double[n];
    q = new double[n];
		
    p[0] =   0.077852054085;
    p[1] =   0.396539319482;
    p[2] =   0.729132090846;
    p[3] =   0.469782287405;
    p[4] = - 0.143906003928;
    p[5] = - 0.224036184993;
    p[6] =   0.071309219266;
    p[7] =   0.080612609151;
    p[8] = - 0.038029936935;
    p[9] = - 0.016574541630;
    p[10] =   0.012550998556;
    p[11] =   0.000429577972;
    p[12] = - 0.001801640704;
    p[13] =   0.000353713799;
		
    for(i=0;i<n;++i) {
      q[i] = p[n-1-i] * ((i-1)%2?-1:1);
    }
  } else if (!strcmp(name.c_str(),"daub16")) {
    n = 16;
    p = new double[n];
    q = new double[n];
		
    p[0] =   0.0544158422431072;
    p[1] =   0.3128715909143166;
    p[2] =   0.6756307362973195;
    p[3] =   0.5853546836542159;
    p[4] = - 0.0158291052563823;
    p[5] = - 0.2840155429615824;
    p[6] =   0.0004724845739124;
    p[7] =   0.1287474266204893;
    p[8] = - 0.0173693010018090;
    p[9] = - 0.0440882539307971;
    p[10] =   0.0139810279174001;
    p[11] =   0.0087460940474065;
    p[12] = - 0.0048703529934520;
    p[13] = - 0.0003917403733770;
    p[14] =   0.0006754494064506;
    p[15] = - 0.0001174767841248;
		
    for(i=0;i<n;++i) {
      q[i] = p[n-1-i] * ((i-1)%2?-1:1);
    }
  } else if (!strcmp(name.c_str(),"daub18")) {
    n = 18;
    p = new double[n];
    q = new double[n];
		
    p[0] =   0.0380779473638778;
    p[1] =   0.2438346746125858;
    p[2] =   0.6048231236900955;
    p[3] =   0.6572880780512736;
    p[4] =   0.1331973858249883;
    p[5] = - 0.2932737832791663;
    p[6] = - 0.0968407832229492;
    p[7] =   0.1485407493381256;
    p[8] =   0.0307256814793385;
    p[9] = - 0.0676328290613279;
    p[10] =   0.0002509471148340;
    p[11] =   0.0223616621236798;
    p[12] = - 0.0047232047577518;
    p[13] = - 0.0042815036824635;
    p[14] =   0.0018476468830563;
    p[15] =   0.0002303857635232;
    p[16] = - 0.0002519631889427;
    p[17] =   0.0000393473203163;
		
    for(i=0;i<n;++i) {
      q[i] = p[n-1-i] * ((i-1)%2?-1:1);
    }
  } else if (!strcmp(name.c_str(),"daub20")) {
    n = 20;
    p = new double[n];
    q = new double[n];
		
    p[0] =   0.0266700579005473;     
    p[1] =   0.1881768000776347;   
    p[2] =   0.5272011889315757;   
    p[3] =   0.6884590394534363;   
    p[4] =   0.2811723436605715;   
    p[5] = - 0.2498464243271598; 
    p[6] = - 0.1959462743772862; 
    p[7] =   0.1273693403357541;  
    p[8] =   0.0930573646035547;  
    p[9] = - 0.0713941471663501; 
    p[10] = - 0.0294575368218399; 
    p[11] =   0.0332126740593612;  
    p[12] =   0.0036065535669870;  
    p[13] = - 0.0107331754833007; 
    p[14] =   0.0013953517470688;  
    p[15] =   0.0019924052951925;  
    p[16] = - 0.0006858566949564; 
    p[17] = - 0.0001164668551285; 
    p[18] =   0.0000935886703202;  
    p[19] = - 0.0000132642028945; 
		
    for(i=0;i<n;++i) {
      q[i] = p[n-1-i] * ((i-1)%2?-1:1);
    }
  } else if (!strcmp(name.c_str(),"daub68")) {
    n = 68;
    p = new double[n];
    q = new double[n];
		
    p[0] =  0.000005770510509196372;   
    p[1] =  0.000129947617286060910; 
    p[2] =  0.001364061360857502900; 
    p[3] =  0.008819889215070597600; 
    p[4] =  0.039048840515836347000; 
    p[5] =  0.124152479453546570000;
    p[6] =  0.287765053073302140000;
    p[7] =  0.478478736036207890000;
    p[8] =  0.530555088298456210000;
    p[9] =  0.290366323291261220000;
    p[10] = -0.128246839428697380000;
    p[11] = -0.331525294405338620000;
    p[12] = -0.103891913282997320000; 
    p[13] =  0.216907215581275150000;
    p[14] =  0.166601746989076480000; 
    p[15] = -0.127337355243722190000; 
    p[16] = -0.160924923462106410000;
    p[17] =  0.077991845108642127000;
    p[18] =  0.134125957419103990000; 
    p[19] = -0.054482963113175950000;
    p[20] = -0.102947593726802140000;
    p[21] =  0.043576097709937658000; 
    p[22] =  0.073185237455317007000; 
    p[23] = -0.037012834328747815000;
    p[24] = -0.047438560026982685000; 
    p[25] =  0.030739754472021749000; 
    p[26] =  0.027228350059302471000;
    p[27] = -0.023671732546529257000;
    p[28] = -0.013143972588625805000; 
    p[29] =  0.016409377976107239000;
    p[30] =  0.004713643252652899500;
    p[31] = -0.010045501052522497000; 
    p[32] = -0.000619476731810230350; 
    p[33] =  0.005334950720477601800;
    p[34] = -0.000769215319245805750; 
    p[35] = -0.002399456109851749600; 
    p[36] =  0.000858994417719352350;
    p[37] =  0.000875199041890761700;
    p[38] = -0.000552735446532307170; 
    p[39] = -0.000232673197326024440;
    p[40] =  0.000265077237852853530;
    p[41] =  0.000026600548344633845; 
    p[42] = -0.000099146977262800837; 
    p[43] =  0.000013531187821126573;
    p[44] =  0.000028449515523895963;
    p[45] = -0.000010576574554128021; 
    p[46] = -0.000005710825840354940;
    p[47] =  0.000004169871888982280;
    p[48] =  0.000000497971843697713; 
    p[49] = -0.000001116306485312163;
    p[50] =  0.000000144819571623039;
    p[51] =  0.000000202599062939615; 
    p[52] = -0.000000075267015756319; 
    p[53] = -0.000000019903464623409;
    p[54] =  0.000000017404232955792; 
    p[55] = -0.000000000866574406854; 
    p[56] = -0.000000002316501897466;
    p[57] =  0.000000000644637807231;
    p[58] =  0.000000000130041029079; 
    p[59] = -0.000000000099047743256;
    p[60] =  0.000000000010042087140;
    p[61] =  0.000000000006080125224; 
    p[62] = -0.000000000002107879064; 
    p[63] =  0.000000000000097994509;
    p[64] =  0.000000000000085791939; 
    p[65] = -0.000000000000023170837; 
    p[66] =  0.000000000000002587338;
    p[67] = -0.000000000000000114894;
		
    for(i=0;i<n;++i) {
      q[i] = p[n-1-i] * ((i-1)%2?-1:1);
    }
  } else if (!strcmp(name.c_str(),"sym4")) {
    n = 4;
    p = new double[n];
    q = new double[n];
		
    p[0] =   0.48296291314469;
    p[1] =   0.83651630373747;
    p[2] =   0.22414386804186;
    p[3] = - 0.12940952255092;
		
    for(i=0;i<n;++i) {
      q[i] = p[n-1-i] * ((i-1)%2?-1:1);
    }
  } else if (!strcmp(name.c_str(),"sym6")) {
    n = 6;
    p = new double[n];
    q = new double[n];
		
    p[0] =   0.33267055295096;
    p[1] =   0.80689150931334;
    p[2] =   0.45987750211933;
    p[3] = - 0.13501102001039;
    p[4] = - 0.08544127388224;
    p[5] =   0.03522629188210;
		
    for(i=0;i<n;++i) {
      q[i] = p[n-1-i] * ((i-1)%2?-1:1);
    }
  } else if (!strcmp(name.c_str(),"sym8")) {
    n = 8;
    p = new double[n];
    q = new double[n];
		
    p[0] =   0.03222310060408;
    p[1] = - 0.01260396726205;
    p[2] = - 0.09921954357696;
    p[3] =   0.29785779560561;
    p[4] =   0.80373875180680;
    p[5] =   0.49761866763256;
    p[6] = - 0.02963552764603;
    p[7] = - 0.07576571478936;
		
    for(i=0;i<n;++i) {
      q[i] = p[n-1-i] * ((i-1)%2?-1:1);
    }
  } else if (!strcmp(name.c_str(),"sym10")) {
    n = 10;
    p = new double[n];
    q = new double[n];
		
    p[0] =   0.01953888273529;
    p[1] = - 0.02110183402476;
    p[2] = - 0.17532808990845;
    p[3] =   0.01660210576452;
    p[4] =   0.63397896345821;
    p[5] =   0.72340769040242;
    p[6] =   0.19939753397739;
    p[7] = - 0.03913424930238;
    p[8] =   0.02951949092577;
    p[9] =   0.02733306834508;
		
    for(i=0;i<n;++i) {
      q[i] = p[n-1-i] * ((i-1)%2?-1:1);
    }
  } else if (!strcmp(name.c_str(),"sym12")) {
    n = 12;
    p = new double[n];
    q = new double[n];
		
    p[0] = - 0.00780070832503;
    p[1] =   0.00176771186424;
    p[2] =   0.04472490177067;
    p[3] = - 0.02106029251230;
    p[4] = - 0.07263752278646;
    p[5] =   0.33792942172762;
    p[6] =   0.78764114103019;
    p[7] =   0.49105594192675;
    p[8] = - 0.04831174258563;
    p[9] = - 0.11799011114819;
    p[10] =   0.00349071208422;
    p[11] =   0.01540410932703;
		
    for(i=0;i<n;++i) {
      q[i] = p[n-1-i] * ((i-1)%2?-1:1);
    }
  } else if (!strcmp(name.c_str(),"sym14")) {
    n = 14;
    p = new double[n];
    q = new double[n];
		
    p[0] =   0.01026817670851;
    p[1] =   0.00401024487153;
    p[2] = - 0.10780823770382;
    p[3] = - 0.14004724044296;
    p[4] =   0.28862963175151;
    p[5] =   0.76776431700316;
    p[6] =   0.53610191709176;
    p[7] =   0.01744125508686;
    p[8] = - 0.04955283493713;
    p[9] =   0.06789269350137;
    p[10] =   0.03051551316596;
    p[11] = - 0.01263630340325;
    p[12] = - 0.00104738488868;
    p[13] =   0.00268181456826;
		
    for(i=0;i<n;++i) {
      q[i] = p[n-1-i] * ((i-1)%2?-1:1);
    }
  } else if (!strcmp(name.c_str(),"sym16")) {
    n = 16;
    p = new double[n];
    q = new double[n];
		
    p[0] =   0.00188995033276;
    p[1] = - 0.00030292051472;
    p[2] = - 0.01495225833705;
    p[3] =   0.00380875201389;
    p[4] =   0.04913717967361;
    p[5] = - 0.02721902991706;
    p[6] = - 0.05194583810771;
    p[7] =   0.36444189483534;
    p[8] =   0.77718575170053;
    p[9] =   0.48135965125838;
    p[10] = - 0.06127335906766;
    p[11] = - 0.14329423835081;
    p[12] =   0.00760748732492;
    p[13] =   0.03169508781149;
    p[14] = - 0.00054213233179;
    p[15] = - 0.00338241595101;
		
    for(i=0;i<n;++i) {
      q[i] = p[n-1-i] * ((i-1)%2?-1:1);
    }
  } else if (!strcmp(name.c_str(),"b4_0.0")) {
    n = 4;
    p = new double[n];
    q = new double[n];
		
    p[0] =   0.48296291;
    p[1] =   0.83651630;
    p[2] =   0.22414387;
    p[3] = - 0.12940952;
		
    for(i=0;i<n;++i) {
      q[i] = p[n-1-i] * ((i-1)%2?-1:1);
    }
  } else if (!strcmp(name.c_str(),"b4_0.1")) {
    n = 4;
    p = new double[n];
    q = new double[n];
		
    p[0] =   0.49123165;
    p[1] =   0.83422448;
    p[2] =   0.21587513;
    p[3] = - 0.12711776;
		
    for(i=0;i<n;++i) {
      q[i] = p[n-1-i] * ((i-1)%2?-1:1);
    }
  } else if (!strcmp(name.c_str(),"b4_0.2")) {
    n = 4;
    p = new double[n];
    q = new double[n];
		
    p[0] =   0.50021916;
    p[1] =   0.83155877;
    p[2] =   0.20688763;
    p[3] = - 0.12445201;
		
    for(i=0;i<n;++i) {
      q[i] = p[n-1-i] * ((i-1)%2?-1:1);
    }
  } else if (!strcmp(name.c_str(),"b4_0.3")) {
    n = 4;
    p = new double[n];
    q = new double[n];
		
    p[0] =   0.50981884;
    p[1] =   0.82850718;
    p[2] =   0.19728794;
    p[3] = - 0.12100415;
		
    for(i=0;i<n;++i) {
      q[i] = p[n-1-i] * ((i-1)%2?-1:1);
    }
  } else if (!strcmp(name.c_str(),"b4_0.4")) {
    n = 4;
    p = new double[n];
    q = new double[n];
		
    p[0] =   0.51995636;
    p[1] =   0.82505202;
    p[2] =   0.18715315;
    p[3] = - 0.11794524;
		
    for(i=0;i<n;++i) {
      q[i] = p[n-1-i] * ((i-1)%2?-1:1);
    }
  } else if (!strcmp(name.c_str(),"b4_0.5")) {
    n = 4;
    p = new double[n];
    q = new double[n];
		
    p[0] =   0.53035917;
    p[1] =   0.82124957;
    p[2] =   0.17674761;
    p[3] = - 0.11414279;
		
    for(i=0;i<n;++i) {
      q[i] = p[n-1-i] * ((i-1)%2?-1:1);
    }
  } else if (!strcmp(name.c_str(),"b4_0.6")) {
    n = 4;
    p = new double[n];
    q = new double[n];
		
    p[0] =   0.54081214;
    p[1] =   0.81716331;
    p[2] =   0.16629464;
    p[3] = - 0.11005653;
		
    for(i=0;i<n;++i) {
      q[i] = p[n-1-i] * ((i-1)%2?-1:1);
    }
  } else if (!strcmp(name.c_str(),"b4_0.7")) {
    n = 4;
    p = new double[n];
    q = new double[n];
		
    p[0] =   0.55103446;
    p[1] =   0.81290209;
    p[2] =   0.15607232;
    p[3] = - 0.10579531;
		
    for(i=0;i<n;++i) {
      q[i] = p[n-1-i] * ((i-1)%2?-1:1);
    }
  } else if (!strcmp(name.c_str(),"b4_0.8")) {
    n = 4;
    p = new double[n];
    q = new double[n];
		
    p[0] =   0.56081569;
    p[1] =   0.80857244;
    p[2] =   0.14629108;
    p[3] = - 0.10146566;
		
    for(i=0;i<n;++i) {
      q[i] = p[n-1-i] * ((i-1)%2?-1:1);
    }
  } else if (!strcmp(name.c_str(),"b4_0.9")) {
    n = 4;
    p = new double[n];
    q = new double[n];
		
    p[0] =   0.57003174;
    p[1] =   0.80426075;
    p[2] =   0.13707504;
    p[3] = - 0.09715397;
		
    for(i=0;i<n;++i) {
      q[i] = p[n-1-i] * ((i-1)%2?-1:1);
    }
  } else if (!strcmp(name.c_str(),"b4_1.0")) {
    n = 4;
    p = new double[n];
    q = new double[n];
		
    p[0] =   0.57860227;
    p[1] =   0.80004303;
    p[2] =   0.12850451;
    p[3] = - 0.09293625;
		
    for(i=0;i<n;++i) {
      q[i] = p[n-1-i] * ((i-1)%2?-1:1);
    }
  } else if (!strcmp(name.c_str(),"b4_1.1")) {
    n = 4;
    p = new double[n];
    q = new double[n];
		
    p[0] =   0.58648244;
    p[1] =   0.79598311;
    p[2] =   0.12062434;
    p[3] = - 0.08887633;
		
    for(i=0;i<n;++i) {
      q[i] = p[n-1-i] * ((i-1)%2?-1:1);
    }
  } else if (!strcmp(name.c_str(),"b4_1.2")) {
    n = 4;
    p = new double[n];
    q = new double[n];
		
    p[0] =   0.59368629;
    p[1] =   0.79211489;
    p[2] =   0.11342048;
    p[3] = - 0.08500810;
		
    for(i=0;i<n;++i) {
      q[i] = p[n-1-i] * ((i-1)%2?-1:1);
    }
  } else if (!strcmp(name.c_str(),"b4_1.3")) {
    n = 4;
    p = new double[n];
    q = new double[n];
		
    p[0] =   0.60026062;
    p[1] =   0.78845055;
    p[2] =   0.10684615;
    p[3] = - 0.08134377;
		
    for(i=0;i<n;++i) {
      q[i] = p[n-1-i] * ((i-1)%2?-1:1);
    }
  } else if (!strcmp(name.c_str(),"b4_1.4")) {
    n = 4;
    p = new double[n];
    q = new double[n];
		
    p[0] =   0.60623768;
    p[1] =   0.78500518;
    p[2] =   0.10086910;
    p[3] = - 0.07789882;
		
    for(i=0;i<n;++i) {
      q[i] = p[n-1-i] * ((i-1)%2?-1:1);
    }
  } else if (!strcmp(name.c_str(),"b4_1.5")) {
    n = 4;
    p = new double[n];
    q = new double[n];
		
    p[0] =   0.61167674;
    p[1] =   0.78177335;
    p[2] =   0.09543005;
    p[3] = - 0.07466658;
		
    for(i=0;i<n;++i) {
      q[i] = p[n-1-i] * ((i-1)%2?-1:1);
    }
  } else if (!strcmp(name.c_str(),"b6_0.0")) {
    n = 6;
    p = new double[n];
    q = new double[n];
		
    p[0] =   0.44841478;
    p[1] =   0.77552168;
    p[2] =   0.39119223;
    p[3] = - 0.14502791;
    p[4] = - 0.13250022;
    p[5] =   0.07661301;
		
    for(i=0;i<n;++i) {
      q[i] = p[n-1-i] * ((i-1)%2?-1:1);
    }
  } else if (!strcmp(name.c_str(),"b6_0.1")) {
    n = 6;
    p = new double[n];
    q = new double[n];
		
    p[0] =   0.45893430;
    p[1] =   0.79242211;
    p[2] =   0.35372518;
    p[3] = - 0.14644658;
    p[4] = - 0.10555270;
    p[5] =   0.06113125;
		
    for(i=0;i<n;++i) {
      q[i] = p[n-1-i] * ((i-1)%2?-1:1);
    }
  } else if (!strcmp(name.c_str(),"b6_0.2")) {
    n = 6;
    p = new double[n];
    q = new double[n];
		
    p[0] =   0.47502467;
    p[1] =   0.80303539;
    p[2] =   0.31496201;
    p[3] = - 0.14495509;
    p[4] = - 0.08287990;
    p[5] =   0.04902648;
		
    for(i=0;i<n;++i) {
      q[i] = p[n-1-i] * ((i-1)%2?-1:1);
    }
  } else if (!strcmp(name.c_str(),"b6_0.3")) {
    n = 6;
    p = new double[n];
    q = new double[n];
		
    p[0] =   0.49303553;
    p[1] =   0.80842688;
    p[2] =   0.27886696;
    p[3] = - 0.14083708;
    p[4] = - 0.06479571;
    p[5] =   0.03951698;
		
    for(i=0;i<n;++i) {
      q[i] = p[n-1-i] * ((i-1)%2?-1:1);
    }
  } else if (!strcmp(name.c_str(),"b6_0.4")) {
    n = 6;
    p = new double[n];
    q = new double[n];
		
    p[0] =   0.51065493;
    p[1] =   0.81006904;
    p[2] =   0.24732487;
    p[3] = - 0.13503181;
    p[4] = - 0.05087302;
    p[5] =   0.03206956;
		
    for(i=0;i<n;++i) {
      q[i] = p[n-1-i] * ((i-1)%2?-1:1);
    }
  } else if (!strcmp(name.c_str(),"b6_0.5")) {
    n = 6;
    p = new double[n];
    q = new double[n];
		
    p[0] = - 0.08338125;
    p[1] =   0.30427515;
    p[2] =   0.84925993;
    p[3] =   0.41893704;
    p[4] = - 0.05877191;
    p[5] = - 0.01610541;
		
    for(i=0;i<n;++i) {
      q[i] = p[n-1-i] * ((i-1)%2?-1:1);
    }
  } else if (!strcmp(name.c_str(),"b6_0.6")) {
    n = 6;
    p = new double[n];
    q = new double[n];
		
    p[0] = - 0.08279787;
    p[1] =   0.30367125;
    p[2] =   0.84917267;
    p[3] =   0.41959533;
    p[4] = - 0.05926801;
    p[5] = - 0.01615980;
		
    for(i=0;i<n;++i) {
      q[i] = p[n-1-i] * ((i-1)%2?-1:1);
    }
  } else if (!strcmp(name.c_str(),"b6_0.7")) {
    n = 6;
    p = new double[n];
    q = new double[n];
		
    p[0] = - 0.08233775;
    p[1] =   0.30257581;
    p[2] =   0.84901846;
    p[3] =   0.42074239;
    p[4] = - 0.05957393;
    p[5] = - 0.01621142;
		
    for(i=0;i<n;++i) {
      q[i] = p[n-1-i] * ((i-1)%2?-1:1);
    }
  } else if (!strcmp(name.c_str(),"b6_0.8")) {
    n = 6;
    p = new double[n];
    q = new double[n];
		
    p[0] = - 0.08195956;
    p[1] =   0.30100677;
    p[2] =   0.84879593;
    p[3] =   0.42236347;
    p[4] = - 0.05972960;
    p[5] = - 0.01626346;
		
    for(i=0;i<n;++i) {
      q[i] = p[n-1-i] * ((i-1)%2?-1:1);
    }
  } else if (!strcmp(name.c_str(),"b6_0.9")) {
    n = 6;
    p = new double[n];
    q = new double[n];
		
    p[0] = - 0.08163489;
    p[1] =   0.29893998;
    p[2] =   0.84849656;
    p[3] =   0.42448474;
    p[4] = - 0.05975489;
    p[5] = - 0.01631794;
		
    for(i=0;i<n;++i) {
      q[i] = p[n-1-i] * ((i-1)%2?-1:1);
    }
  } else if (!strcmp(name.c_str(),"b6_1.0")) {
    n = 6;
    p = new double[n];
    q = new double[n];
		
    p[0] = - 0.08134182;
    p[1] =   0.29631466;
    p[2] =   0.84810454;
    p[3] =   0.42716837;
    p[4] = - 0.05965593;
    p[5] = - 0.01637625;
		
    for(i=0;i<n;++i) {
      q[i] = p[n-1-i] * ((i-1)%2?-1:1);
    }
  } else if (!strcmp(name.c_str(),"b6_1.1")) {
    n = 6;
    p = new double[n];
    q = new double[n];
		
    p[0] = - 0.08106069;
    p[1] =   0.29303272;
    p[2] =   0.84759506;
    p[3] =   0.43051332;
    p[4] = - 0.05942758;
    p[5] = - 0.01643926;
		
    for(i=0;i<n;++i) {
      q[i] = p[n-1-i] * ((i-1)%2?-1:1);
    }
  } else if (!strcmp(name.c_str(),"b6_1.2")) {
    n = 6;
    p = new double[n];
    q = new double[n];
		
    p[0] = - 0.08077071;
    p[1] =   0.28895499;
    p[2] =   0.84693143;
    p[3] =   0.43465896;
    p[4] = - 0.05905393;
    p[5] = - 0.01650717;
		
    for(i=0;i<n;++i) {
      q[i] = p[n-1-i] * ((i-1)%2?-1:1);
    }
  } else if (!strcmp(name.c_str(),"b6_1.3")) {
    n = 6;
    p = new double[n];
    q = new double[n];
		
    p[0] = - 0.08044643;
    p[1] =   0.28389820;
    p[2] =   0.84606093;
    p[3] =   0.43978754;
    p[4] = - 0.05850772;
    p[5] = - 0.01657896;
		
    for(i=0;i<n;++i) {
      q[i] = p[n-1-i] * ((i-1)%2?-1:1);
    }
  } else if (!strcmp(name.c_str(),"b6_1.4")) {
    n = 6;
    p = new double[n];
    q = new double[n];
		
    p[0] = - 0.08005385;
    p[1] =   0.27764083;
    p[2] =   0.84491061;
    p[3] =   0.44611735;
    p[4] = - 0.05774997;
    p[5] = - 0.01665140;
		
    for(i=0;i<n;++i) {
      q[i] = p[n-1-i] * ((i-1)%2?-1:1);
    }
  } else if (!strcmp(name.c_str(),"b6_1.5")) {
    n = 6;
    p = new double[n];
    q = new double[n];
		
    p[0] = - 0.07954693;
    p[1] =   0.26995392;
    p[2] =   0.84338655;
    p[3] =   0.45387025;
    p[4] = - 0.05673284;
    p[5] = - 0.01671738;
		
    for(i=0;i<n;++i) {
      q[i] = p[n-1-i] * ((i-1)%2?-1:1);
    }
  } else if (!strcmp(name.c_str(),"beylkin18")) {
    n = 18;
    p = new double[n];
    q = new double[n];
		
    p[0] =   0.0993057653743539270;
    p[1] =   0.424215360812961410;
    p[2] =   0.699825214056600590;
    p[3] =   0.449718251149468670;
    p[4] = - 0.110927598348234300;
    p[5] = - 0.264497231446384820;
    p[6] =   0.0269003088036903200;
    p[7] =   0.155538731877093800;
    p[8] = - 0.0175207462665296490;
    p[9] = - 0.0885436306229248350;
    p[10] =   0.0196798660443221200;
    p[11] =   0.0429163872741922730;
    p[12] = - 0.0174604086960288290;
    p[13] = - 0.0143658079688526110;
    p[14] =   0.0100404118446319900;
    p[15] =   0.00148423478247234610;
    p[16] = - 0.00273603162625860610;
    p[17] =   0.000640485328521245350;
		
    for(i=0;i<n;++i) {
      q[i] = p[n-1-i] * ((i-1)%2?-1:1);
    }
  } else if (!strcmp(name.c_str(),"coif6")) {
    n = 6;
    p = new double[n];
    q = new double[n];
		
    p[0] = - 0.07273261951285;
    p[1] =   0.33789766245781;
    p[2] =   0.85257202021226;
    p[3] =   0.38486484686420;
    p[4] = - 0.07273261951285;
    p[5] = - 0.01565572813546;
		
    for(i=0;i<n;++i) {
      q[i] = p[n-1-i] * ((i-1)%2?-1:1);
    }
  } else if (!strcmp(name.c_str(),"coif12")) {
    n = 12;
    p = new double[n];
    q = new double[n];
		
    p[0] =   0.01638733646360;
    p[1] = - 0.04146493678197;
    p[2] = - 0.06737255472230;
    p[3] =   0.38611006682309;
    p[4] =   0.81272363544961;
    p[5] =   0.41700518442378;
    p[6] = - 0.07648859907869;
    p[7] = - 0.05943441864675;
    p[8] =   0.02368017194645;
    p[9] =   0.00561143481942;
    p[10] = - 0.00182320887071;
    p[11] = - 0.00072054944537;
		
    for(i=0;i<n;++i) {
      q[i] = p[n-1-i] * ((i-1)%2?-1:1);
    }
  } else if (!strcmp(name.c_str(),"coif18")) {
    n = 18;
    p = new double[n];
    q = new double[n];
		
    p[0] = - 0.00379351286449;
    p[1] =   0.00778259642733; 
    p[2] =   0.02345269614184; 
    p[3] = - 0.06577191128186;
    p[4] = - 0.06112339000267;
    p[5] =   0.40517690240962; 
    p[6] =   0.79377722262562; 
    p[7] =   0.42848347637762; 
    p[8] = - 0.07179982161931;
    p[9] = - 0.08230192710689;
    p[10] =   0.03455502757306;
    p[11] =   0.01588054486362;
    p[12] = - 0.00900797613666;
    p[13] = - 0.00257451768875;
    p[14] =   0.00111751877089;
    p[15] =   0.00046621696011;
    p[16] = - 0.00007098330314;
    p[17] = - 0.00003459977284;
		
    for(i=0;i<n;++i) {
      q[i] = p[n-1-i] * ((i-1)%2?-1:1);
    }
  }  else if (!strcmp(name.c_str(),"coif24")) {
    n = 24;
    p = new double[n];
    q = new double[n];
		
    p[0] =   0.00089231366858;     
    p[1] = - 0.00162949201260;   
    p[2] = - 0.00734616632764;   
    p[3] =   0.01606894396478;   
    p[4] =   0.02668230015605;   
    p[5] = - 0.08126669968088; 
    p[6] = - 0.05607731331675; 
    p[7] =   0.41530840703043;  
    p[8] =   0.78223893092050;  
    p[9] =   0.43438605649147; 
    p[10] = - 0.06662747426343; 
    p[11] = - 0.09622044203399;  
    p[12] =   0.03933442712334;  
    p[13] =   0.02508226184486; 
    p[14] = - 0.01521173152795;  
    p[15] = - 0.00565828668661;  
    p[16] =   0.00375143615728; 
    p[17] =   0.00126656192930; 
    p[18] = - 0.00058902075624;  
    p[19] = - 0.00025997455249; 
    p[20] =   0.00006233903446;
    p[21] =   0.00003122987587;
    p[22] = - 0.00000325968024;
    p[23] = - 0.00000178498500;
		
    for(i=0;i<n;++i) {
      q[i] = p[n-1-i] * ((i-1)%2?-1:1);
    }
  }  else if (!strcmp(name.c_str(),"coif30")) {
    n = 30;
    p = new double[n];
    q = new double[n];
		
    p[0] = - 0.00021208083983;     
    p[1] =   0.00035858968793;   
    p[2] =   0.00217823635833;   
    p[3] = - 0.00415935878180;   
    p[4] = - 0.01013111752086;   
    p[5] =   0.02340815678818; 
    p[6] =   0.02816802897375; 
    p[7] = - 0.09192001056889;  
    p[8] = - 0.05204316318145;  
    p[9] =   0.42156620673301; 
    p[10] =   0.77428960373039; 
    p[11] =   0.43799162621564;  
    p[12] = - 0.06203596396911;  
    p[13] = - 0.10557420871390; 
    p[14] =   0.04128920875431;  
    p[15] =   0.03268357427038;  
    p[16] = - 0.01976177894455; 
    p[17] = - 0.00916423116340; 
    p[18] =   0.00676418544873;  
    p[19] =   0.00243337321290; 
    p[20] = - 0.00166286370218;
    p[21] = - 0.00063813134311;
    p[22] =   0.00030225958184;
    p[23] =   0.00014054114972;
    p[24] = - 0.00004134043228;
    p[25] = - 0.00002131502681;
    p[26] =   0.00000373465518;
    p[27] =   0.00000206376185;
    p[28] = - 0.00000016744289;
    p[29] = - 0.00000009517657;
		
    for(i=0;i<n;++i) {
      q[i] = p[n-1-i] * ((i-1)%2?-1:1);
    }
  } else if (!strcmp(name.c_str(),"astro")) {
    n = 5;
    p = new double[n];
    q = new double[n];
		
    p[0] =   1.0/16.0;
    p[1] =   4.0/16.0;
    p[2] =   6.0/16.0;
    p[3] =   4.0/16.0;
    p[4] =   1.0/16.0;
		
    q[0] = - 1.0/16.0;
    q[1] = - 4.0/16.0;
    q[2] =   10.0/16.0;
    q[3] = - 4.0/16.0;
    q[4] = - 1.0/16.0;
		
  } 
	
  *fsize = n;
  *P = p;
  *Q = q;
	
  return;
}

void lts2::WaveletCoef_Biortho(const std::string& name,double **P,double **Q,int *fsizeP, int *fsizeQ) {
	
  double *p = NULL;
  double *q = NULL;
  double sq;
	
  free(*P);
  free(*Q);
	
  if (!strcmp(name.c_str(),"9-7") || !strcmp(name.c_str(),"9/7")) {
    *fsizeP = 9;
    *fsizeQ = 7;
		
    p = new double[*fsizeP];
    q = new double[*fsizeQ];
		
    p[0] =   0.02674875741081;
    p[1] =  -0.01686411844287;
    p[2] =  -0.07822326652899;
    p[3] =   0.26686411844287;
    p[4] =   0.60294901823636;
    p[5] =   0.26686411844287;
    p[6] =  -0.07822326652899;
    p[7] =  -0.01686411844287;
    p[8] =   0.02674875741081;
		
    q[0] =   0.04563588155713;
    q[1] =  -0.02877176311425;
    q[2] =  -0.29563588155713;
    q[3] =   0.55754352622850;
    q[4] =  -0.29563588155713;
    q[5] =  -0.02877176311425;
    q[6] =   0.04563588155713;
		
    /*
      p[0] = 0.037828455506995; 
      p[1] = -0.023849465019380;
      p[2] = -0.11062440441842;
      p[3] = 0.37740285561265;
      p[4] = 0.85269867900940;
      p[5] = 0.37740285561265;
      p[6] = -0.11062440441842;
      p[7] = -0.023849465019380;
      p[8] = 0.037828455506995;
		 
      q[0] = -0.064538882628938;
      q[1] = -0.040689417609558;
      q[2] = 0.41809227322221;
      q[3] = 0.78848561640566;
      q[4] = 0.41809227322221;
      q[5] = -0.040689417609558;
      q[6] = -0.064538882628938;
    */
		
  } else if (!strcmp(name.c_str(),"5-3") || !strcmp(name.c_str(),"5/3")) {
    *fsizeP = 5;
    *fsizeQ = 3;
		
    p = new double[5];
    q = new double[3];
		
    sq = 2*sqrt(2);  
		
    p[0] = p[4] = -1/(2*sq);
    p[1] = p[3] = q[0] = q[2] = 1/sq;
    p[2] = 3/sq;
		
    q[1] = 2/sq;
  }
	
  *P = p;
  *Q = q;
	
  return;
}

void lts2::WaveletTransform2D(const cv::Mat& src, cv::Mat& dest, 
                              double *P, double *Q, int fsize, int *ww, int *wh, int stop_after)
{
  if (!src.data) 
  {
    std::cerr << "gmg_WaveletTransform2D() src == NULL" << std::endl;
    return;
  }
	
  cv::Size imageSize = src.size();
    
  int input_w = imageSize.width;
  int input_h = imageSize.height;
	
  if (stop_after < 0) 
    stop_after = MAX(input_w, input_h);
	
  dest.create(imageSize, src.type());
	
  int width = imageSize.width;//XBDW(src)-XBUP(src);
  int height = imageSize.height;//YBDW(src)-YBUP(src);
	
  std::vector<cv::Mat> components;
  cv::split(src, components);
	
  int loop_count = 0;
  while(width>=fsize && height>=fsize && !(width%2) && !(height%2) && loop_count < stop_after) 
  {
		
    for(int b = 0; b < components.size(); ++b) 
    {
      for(int k = 0; k < height; ++k)
        Wavegeneral_h(components.at(b), k, width, 0, P, Q, fsize);
      for(int k = 0; k < width; ++k)
        Wavegeneral_v(components.at(b), k, height, 0, P, Q, fsize);
    }
        
    *wh = height;
    *ww = width;
		
    width /= 2; 
    height /= 2;
		
    ++loop_count;    
  }
	
  cv::merge(components, dest);
}


void lts2::InverseWaveletTransform2D(const cv::Mat& src, cv::Mat& dest, double *P, double *Q, int fsize, int *ww, int *wh)
{
  if (!src.data) {
    std::cerr << "InverseWaveletTransform2D() src == NULL" << std::endl;
    return;
  }
	
  cv::Size imageSize = src.size();
    
  dest.create(imageSize, src.type());
	
//	int width = imageSize.width; //XBDW(src)-XBUP(src);
//	int height = imageSize.height; //YBDW(src)-YBUP(src);
	
  std::vector<cv::Mat> components;
  cv::split(src, components);
	
  int width = *ww; 
  int height = *wh;
	
  if (width == 0 || height == 0)
    return;
    
  while(width <= imageSize.width && height <= imageSize.height) 
  {
    for(int b = 0; b < components.size(); ++b) 
    {  
      for(int k = 0; k < width; ++k)
        Wavegeneral_v(components.at(b), k, height, -1, P, Q, fsize);
      for(int k = 0; k < height; ++k)
        Wavegeneral_h(components.at(b), k, width, -1, P, Q, fsize);
    }
    width *= 2;
    height *= 2;
  }
    
  cv::merge(components, dest);
}

void lts2::ImageWaveletScale(const cv::Mat& src, cv::Mat& dest, double *P, double *Q, int fsize, int scale, int *rscale)
{
  cv::Mat ret;
  float *min, *max, a, c;
	
  cv::Size imageSize = src.size();
        
  std::vector<cv::Mat> components, dest_components;
  cv::split(src, components);
    
  double *omin = new double[components.size()];
  double *omax = new double[components.size()];
    
  for (int b = 0; b < components.size(); ++b)
    cv::minMaxLoc(components.at(b), &omin[b], &omax[b]);
	
  int width = src.cols;
  int height = src.rows;
  int num = 1;
	
  while(num<=scale && width>=fsize && height>=fsize && !(width%2) && !(height%2)) 
  {
    for (int b = 0; b < components.size(); ++b) 
    {
      for(int k = 0; k < height; ++k)
        Wavegeneral_h(components.at(b), k, width, 0, P, Q, fsize);
      for(int k = 0; k < width; ++k)
        Wavegeneral_v(components.at(b), k, height, 0, P, Q, fsize);
    }
    width /= 2; 
    height /= 2;
    ++num;
  }
    
  dest_components.reserve(components.size());
    	
  min = new float[components.size()];
  max = new float[components.size()];
    
  for (int b = 0; b < components.size(); ++b) 
  {
    min[b] = max[b] = components.at(b).at<float>(0, 0);
    dest_components.at(b).create(imageSize, CV_32FC1);
        
    for(int j = 0; j < height; ++j) 
    {
      for(int i = 0; i < width; ++i) 
      {
        min[b] = MIN(components.at(b).at<float>(j, i), min[b]);
        max[b] = MAX(components.at(b).at<float>(j, i), max[b]);
      }
    }
    a = (omax[b] - omin[b]) / (max[b] - min[b]);
    c = omin[b] - min[b]*(omax[b] - omin[b]) / (max[b] - min[b]);
        
    dest_components.at(b) = a * components.at(b) + c;
  }
    
  cv::merge(dest_components, dest);
    
  if (scale != 0) 
    *rscale = num - 1;
  else 
    *rscale = 0;
    
  delete[] min;
  delete[] max;
  delete[] omin;
  delete[] omax;
	
  return;
}

void lts2::CreateWaveletDisplay(const cv::Mat &waves, cv::Mat &display, int ww, int wh)
{
  display.create(waves.size(), CV_8U);
  display.setTo(cv::Scalar(0));
    
  // Find the continuous part first
  int subWindowWidth = ww / 2;
  int subWindowHeight = wh / 2;

  cv::Mat submat;
    
  cv::Mat C;
  submat = waves(cv::Range(0, subWindowHeight), cv::Range(0, subWindowWidth));
    
  double wmin, wmax, wrange;
  cv::minMaxLoc(submat, &wmin, &wmax);
  wrange = wmax - wmin;
    
  submat.convertTo(C, CV_8UC1, 255.0/wrange, -255.0f*wmin/wrange);
    
  for (int y = 0; y < subWindowHeight; ++y)
  {
    const uchar* srcptr = C.ptr<uchar>(y);
    uchar* destptr = display.ptr<uchar>(y);
        
    for (int x = 0; x < subWindowWidth; ++x)
      *destptr++ = *srcptr++;
  }
    
  // Now for the diff images
  int offsetX = subWindowWidth;
  int offsetY = subWindowHeight;
    
  while (subWindowWidth <= waves.cols/2 && subWindowHeight <= waves.rows/2) 
  {
    int maxX = MIN(waves.cols, offsetX + subWindowWidth);
    int maxY = MIN(waves.rows, offsetY + subWindowHeight);
        
    // HH
    cv::Mat HH = waves(cv::Range(0, maxY), cv::Range(offsetX, maxX));
    cv::minMaxLoc(HH, &wmin, &wmax);
    wrange = wmax - wmin;
    HH.convertTo(C, CV_8UC1, 255.0/wrange, -255.0*wmin/(wmax-wmin));
        
    for (int y = 0; y < subWindowHeight; ++y)
    {
      const uchar* srcptr = C.ptr<uchar>(y);
      uchar* destptr = display.ptr<uchar>(y) + offsetX;
            
      for (int x = 0; x < subWindowWidth; ++x)
        *destptr++ = *srcptr++;
    }

    // VV
    cv::Mat VV = waves(cv::Range(offsetY, maxY), cv::Range(0, maxX));
    cv::minMaxLoc(VV, &wmin, &wmax);
    wrange = wmax - wmin;
    VV.convertTo(C, CV_8UC1, 255.0/wrange, -255.0*wmin/(wmax-wmin));
        
    for (int y = 0; y < subWindowHeight; ++y)
    {
      const uchar* srcptr = C.ptr<uchar>(y);
      uchar* destptr = display.ptr<uchar>(y+offsetY);
            
      for (int x = 0; x < subWindowWidth; ++x)
        *destptr++ = *srcptr++;
    }
        
    // HV
    cv::Mat HV = waves(cv::Range(offsetY, maxY), cv::Range(offsetX, maxX));
    cv::minMaxLoc(HV, &wmin, &wmax);
    wrange = wmax - wmin;
    HV.convertTo(C, CV_8UC1, 255.0/wrange, -255.0*wmin/(wmax-wmin));
        
    for (int y = 0; y < subWindowHeight; ++y)
    {
      const uchar* srcptr = C.ptr<uchar>(y);
      uchar* destptr = display.ptr<uchar>(y+offsetY) + offsetX;
            
      for (int x = 0; x < subWindowWidth; ++x)
        *destptr++ = *srcptr++;
    }
        
    // Update the bounds
    offsetX += subWindowWidth;
    offsetY += subWindowHeight;
    subWindowWidth *= 2;
    subWindowHeight *= 2;
  }
}

void lts2::HardThresholding(const cv::Mat &src, cv::Mat &thresholdedCoeffs, float threshold)
{
  src.copyTo(thresholdedCoeffs);
    
  int rows = thresholdedCoeffs.rows;
  int cols = thresholdedCoeffs.cols;
    
  if (thresholdedCoeffs.isContinuous())
  {
    cols *= rows;
    rows = 1;
  }
    
  for (int y = 0; y < rows; ++y)
  {
    float *p_coeffs = thresholdedCoeffs.ptr<float>(y);
    for (int x = 0; x < cols; ++x) 
      *p_coeffs *= (std::fabs(*p_coeffs) - threshold > 0.0f ? 1.0f: 0.0f);
  }
}

void lts2::SoftThresholding(const cv::Mat &src, cv::Mat &thresholdedCoeffs, float threshold)
{
  cv::Mat tmp;
  tmp = cv::abs(src) - threshold;
    
  thresholdedCoeffs.create(src.size(), CV_32F);
    
  for (int y = 0; y < src.rows; ++y)
  {
    const float *p_src = src.ptr<float>(y);
    const float *p_tsrc = tmp.ptr<float>(y);
    float *p_th = thresholdedCoeffs.ptr<float>(y);
        
    for (int x = 0; x < src.cols; ++x, ++p_tsrc, ++p_src)
    {
      if (*p_tsrc <= 0.0f)
        *p_th++ = 0.0f;
      else
      {
        float sign = (*p_src > 0.0f) ? 1.0f: -1.0f;
        *p_th++ = sign * *p_tsrc;
      }
    }
  }
}

void lts2::SoftThresholdingPreserveContinuous(const cv::Mat &src, cv::Mat &thresholdedCoeffs, float threshold)
{
  cv::Mat tmp;
  tmp = cv::abs(src) - threshold;
    
  thresholdedCoeffs.create(src.size(), CV_32F);

  int rows = src.rows;
  int cols = src.cols;

  int minRow = rows / 2;
  int minCol = cols / 2;
    
  for (int y = 0; y < rows; ++y)
  {
    const float *p_src = src.ptr<float>(y);
    const float *p_tsrc = tmp.ptr<float>(y);
    float *p_th = thresholdedCoeffs.ptr<float>(y);
        
    for (int x = 0; x < cols; ++x, ++p_tsrc, ++p_src)
    {
      if (y >= minRow || x >= minCol)
      {
        float sign = 1.0f;
        if (*p_src < 0.0f) sign = -1.0f;
                
        *p_th++ = sign * (*p_tsrc) * (*p_tsrc > 0.0f);
      }
      else *p_th++ = *p_src;
    }
  }
}

void lts2::KillDiagonalWaveletCoefficients(const cv::Mat &src, cv::Mat &cleaned, int ww, int wh)
{
  src.copyTo(cleaned);
     
  int minY = wh / 2;
  int minX = ww / 2;
    
  ww /= 2;
  wh /= 2;
    
  while (ww <= src.cols/2 && wh <= src.rows/2)
  {
    int maxX = MIN(src.cols, minX + ww);
    int maxY = MIN(src.rows, minY + wh);
        
    for (int y = minY; y < maxY; ++y)
    {
      float *p_clean = cleaned.ptr<float>(y) + minX;
            
      for (int x = minX; x < maxX; ++x)
        *p_clean++ = 0.0f;
    }
        
    minX = maxX;
    minY = maxY;
        
    ww *= 2;
    wh *= 2;
  }
}

void lts2::WaveletTransform(cv::Mat const &X, int Jmin, std::string const &wavelet, unsigned int options)
{
  float const h[] = {1.586134342, -.05298011854, -.8829110762, .4435068522, 1.149604398};
  bool isReverse = false;
  bool isTranslationInvariant = false;
    
  //---------------- Handle separable wavelet transform here ----------------
    
  //---------------- Non-separable case ----------------
    
  int n = X.rows;
  int m = (sizeof(h)/sizeof(float) - 1)/2;
  int Jmax = (int)floor(log2(n)) - 1;

  std::vector<int> Jlist;
  int j = Jmax;
  while (j > Jmin) 
  {
    Jlist.push_back(j);
    --j;
  }
  if (isReverse)
  {
    std::reverse(Jlist.begin(), Jlist.end());
  }
    
  if (isTranslationInvariant == false)
  {
    std::vector<int>::const_iterator it_j = Jlist.begin();
    for ( ; it_j != Jlist.end(); ++it_j)
    {
      // Horizontal lifting step
      // Vertical lifting step
    }
  }
  else
  {
        
  }
}
