#!/usr/bin/env wolframscript
(* ::Package:: *)

path = $ScriptCommandLine[[2]];
myNBpath=SetDirectory[path];
images=FileNames["*.png"];
Print[images];
HistogramTransformationWeightedLocalAdaptive[img_,radius_]:=Module[
{funs,nodes,w,h,f,fun,fi},funs=Map[HistogramTransformInterpolation,ImagePartition[img,{2*radius+1}],{2}];
{w,h}=ImageDimensions@img;
nodes=Map[Mean@Flatten[N@ImageData[#,Automatic],1]&,ImagePartition[Image[Array[{#1,#2}&,{h,w}],"Bit16"],{2*radius+1}],{2}];
nodes=ArrayPad[nodes,{{1,1},{1,1}},"Fixed"];
nodes[[All,{1,-1},2]]={1,w};
nodes[[{1,-1},All,1]]={1,h};
funs=ArrayPad[funs,1,"Fixed"];
fi=Interpolation[Thread[{Round@Flatten[nodes,1],Table[f[#],{f,Flatten@funs}]}],InterpolationOrder->1];
fun[v_,{r_,c_}]:=Evaluate[fi[r,c]]&[v];
Quiet@ImageApplyIndexed[fun,img]];

ApplyHist[filename_] := Module[{name = filename},
img = Import[name];
{r, g, b} = ColorSeparate[img];
radius = 10;
Timing[r=HistogramTransformationWeightedLocalAdaptive[r,radius]];
Timing[g=HistogramTransformationWeightedLocalAdaptive[g,radius]];
Timing[b=HistogramTransformationWeightedLocalAdaptive[b,radius]];
img  = ColorCombine[{r, g, b}];
Export[StringJoin["norm_", filename], img, "PNG" ];
];

ApplyHist/@FileNames["*.png"]


