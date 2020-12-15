/* Qtheory*/
/* Author: Jiachun Wang*/

%let wrds=wrds.wharton.upenn.edu 4016;
options comamid=TCP remote=WRDS;
signon username=_prompt_;

rsubmit;

LIBNAME famamac '~/jcw/FamaMacBeth';
LIBNAME macro '~/jcw/Macros';
LIBNAME tempo '~/jcw/Temp';

%include '~/jcw/Macros/Fama_MacBeth.sas'; run;
%include '~/jcw/Macros/winsor_macro.sas'; run;

%let BEGDATE=01JAN2000;
%let ENDDATE=25JUL2018;
 
/* Step1. Extract Compustat Sample */
data comp1;
set comp.funda (keep = gvkey datadate sich indfmt datafmt popsrc consol fyear fyr
 SEQ PSTKRV PSTKL PSTK TXDB ITCB PRCC_C CSHO AT);
where datadate between "&BEGDATE"d and "&ENDDATE"d
 and DATAFMT='STD' and INDFMT='INDL' and CONSOL='C' and POPSRC='D'
 and sich between 4900 and 4999;
/* Use Daniel and Titman (JF 1997) Book of Equity Calculation: */
if SEQ>0; /* Keep Companies with Existing Shareholders' Equity */
/* PSTKRV: Preferred stock Redemption Value . If missing, use PSTKL: Liquidating Value */
/* If still missing, then use PSTK: Preferred stock - Carrying Value, Stock (Capital)  */
PREF = coalesce(PSTKRV,PSTKL,PSTK);
/* BE = Stockholders Equity + Deferred Taxes + Investment Tax Credit - Preferred Stock */
BE = sum(SEQ, TXDB, ITCB, -PREF);
/* Calculate Market Value of Equity at Year End */
/* use prrc_c at the calendar year end for a fair cross sectional comparison */
ME = PRCC_C*CSHO;
/* Set missing retained earnings and missing current assets figures to zero */
if missing(RE) then RE=0; if missing(ACT) then ACT=0;
/* Calculate Tobin's Q */
Tobin_Q = (AT + ME - BE) / AT;
label datadate = "Fiscal Year End Date";
label BE = "Book Value of Equity, x$mil";
label ME = "Market Value of Equity, x$mil";
label Tobin_Q ="Tobin's Q";
format AT BE ME dollar12.3 Tobin_Q comma12.2;
keep GVKEY sich datadate fyear fyr be me Tobin_Q;
run;
 
/* Step2. Get monthly return and add gvkey-permno link*/
proc sql;
  create table crsp1 (drop=LinkScore) 
  as select distinct a.permno, a.date, a.ret, b.gvkey, 
   case 
    when b.linkprim='P' then 2 when b.linkprim='C' then 1 else 0 
   end as LinkScore
  from crsp.msf as a, crsp.ccmxpf_linktable as b
  where a.permno = b.lpermno and
  b.LINKTYPE in ("LU","LC") and 
 (b.LINKDT <= a.date) and (a.date <= b.LINKENDDT or missing(b.LINKENDDT))
group by a.permno, a.date
having LinkScore=max(LinkScore);
quit;


/* Step3. Adding fundas */

proc sql;
create table crsp2
as select a.permno, a.date, a.ret, b.*
from crsp1 as a, comp1 as b
where a.date between "&BEGDATE"d and "&ENDDATE"d 
  and a.gvkey = b.gvkey 
  /* Match all available Tobin's Q before the date of observation in CRSP*/
  and b.datadate<=a.date 
order by permno, date;
quit;

 /* sort date of available Tobin's Q descendingly*/
proc sort data=crsp2; by permno date descending datadate; run;

 /* Since nodupley keep the first obs, thus the most most recent Tobin�s q available is kept*/
proc sort data=crsp2 nodupkey; by permno date; run;

/* Provide a summary statistics table of original data*/
proc means data = crsp2 N Mean Min P25 Median P75 Max     ; 
	var ret Tobin_Q;
run;

/* The summary statistics does not show very odd value */
/* But I winsorize the data at 0.5 level just to be cautious */
%winsor(dsetin=crsp2, dsetout=crsp2_w, byvar=none, 
vars= ret Tobin_Q, type=winsor, pctl=0.5 99.5);

/* Provide a summary statistics table of winsorized data*/
proc means data = crsp2_w N Mean Min P25 Median P75 Max     ; 
	var ret Tobin_Q;
run;

data crsp3; set crsp2_w;
 	format date yymms7.;
run;

data tempo.qtheory_temp; set crsp3; run;
/* Filter out firms with Tobin's Q lower/higher than 1*/
data crsp3_high; set crsp3;
if Tobin_Q<1 then delete;
run;
data crsp3_low; set crsp3;
if Tobin_Q>1 then delete;
run;

/* Running Fama-MacBeth regression for three set of data*/
%FM(inset=crsp3, outset=result, datevar=date, depvar=ret, indvars=Tobin_Q, lag=1)
%FM(inset=crsp3_low, outset=result2, datevar=date, depvar=ret, indvars=Tobin_Q, lag=1)
%FM(inset=crsp3_high, outset=result3, datevar=date, depvar=ret, indvars=Tobin_Q, lag=1)

/* Print out result */
proc print data=result;run;
proc print data=result2;run;
proc print data=result3;run;

endrsubmit;



