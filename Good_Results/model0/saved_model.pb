±!
??
B
AssignVariableOp
resource
value"dtype"
dtypetype?
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
?
Conv3D

input"T
filter"T
output"T"
Ttype:
2"
strides	list(int)(0""
paddingstring:
SAMEVALID"0
data_formatstringNDHWC:
NDHWCNCDHW"!
	dilations	list(int)	

.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
?
	MaxPool3D

input"T
output"T"
ksize	list(int)(0"
strides	list(int)(0""
paddingstring:
SAMEVALID"0
data_formatstringNDHWC:
NDHWCNCDHW"
Ttype:
2
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype?
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
?
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ?
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.4.12v2.4.0-49-g85c8b2a817f8??
?
conv3d_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:d* 
shared_nameconv3d_3/kernel

#conv3d_3/kernel/Read/ReadVariableOpReadVariableOpconv3d_3/kernel**
_output_shapes
:d*
dtype0
r
conv3d_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv3d_3/bias
k
!conv3d_3/bias/Read/ReadVariableOpReadVariableOpconv3d_3/bias*
_output_shapes
:*
dtype0
?
conv3d_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
* 
shared_nameconv3d_4/kernel

#conv3d_4/kernel/Read/ReadVariableOpReadVariableOpconv3d_4/kernel**
_output_shapes
:
*
dtype0
r
conv3d_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*
shared_nameconv3d_4/bias
k
!conv3d_4/bias/Read/ReadVariableOpReadVariableOpconv3d_4/bias*
_output_shapes
:
*
dtype0
{
dense_12/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?* 
shared_namedense_12/kernel
t
#dense_12/kernel/Read/ReadVariableOpReadVariableOpdense_12/kernel*
_output_shapes
:	?*
dtype0
r
dense_12/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_12/bias
k
!dense_12/bias/Read/ReadVariableOpReadVariableOpdense_12/bias*
_output_shapes
:*
dtype0
{
dense_13/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?* 
shared_namedense_13/kernel
t
#dense_13/kernel/Read/ReadVariableOpReadVariableOpdense_13/kernel*
_output_shapes
:	?*
dtype0
r
dense_13/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_13/bias
k
!dense_13/bias/Read/ReadVariableOpReadVariableOpdense_13/bias*
_output_shapes
:*
dtype0
{
dense_14/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?* 
shared_namedense_14/kernel
t
#dense_14/kernel/Read/ReadVariableOpReadVariableOpdense_14/kernel*
_output_shapes
:	?*
dtype0
r
dense_14/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_14/bias
k
!dense_14/bias/Read/ReadVariableOpReadVariableOpdense_14/bias*
_output_shapes
:*
dtype0
{
dense_15/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?* 
shared_namedense_15/kernel
t
#dense_15/kernel/Read/ReadVariableOpReadVariableOpdense_15/kernel*
_output_shapes
:	?*
dtype0
r
dense_15/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_15/bias
k
!dense_15/bias/Read/ReadVariableOpReadVariableOpdense_15/bias*
_output_shapes
:*
dtype0
{
dense_16/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?* 
shared_namedense_16/kernel
t
#dense_16/kernel/Read/ReadVariableOpReadVariableOpdense_16/kernel*
_output_shapes
:	?*
dtype0
r
dense_16/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_16/bias
k
!dense_16/bias/Read/ReadVariableOpReadVariableOpdense_16/bias*
_output_shapes
:*
dtype0
{
dense_17/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?* 
shared_namedense_17/kernel
t
#dense_17/kernel/Read/ReadVariableOpReadVariableOpdense_17/kernel*
_output_shapes
:	?*
dtype0
r
dense_17/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_17/bias
k
!dense_17/bias/Read/ReadVariableOpReadVariableOpdense_17/bias*
_output_shapes
:*
dtype0
{
dense_18/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?* 
shared_namedense_18/kernel
t
#dense_18/kernel/Read/ReadVariableOpReadVariableOpdense_18/kernel*
_output_shapes
:	?*
dtype0
r
dense_18/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_18/bias
k
!dense_18/bias/Read/ReadVariableOpReadVariableOpdense_18/bias*
_output_shapes
:*
dtype0
{
dense_19/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?* 
shared_namedense_19/kernel
t
#dense_19/kernel/Read/ReadVariableOpReadVariableOpdense_19/kernel*
_output_shapes
:	?*
dtype0
r
dense_19/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_19/bias
k
!dense_19/bias/Read/ReadVariableOpReadVariableOpdense_19/bias*
_output_shapes
:*
dtype0
{
dense_20/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?* 
shared_namedense_20/kernel
t
#dense_20/kernel/Read/ReadVariableOpReadVariableOpdense_20/kernel*
_output_shapes
:	?*
dtype0
r
dense_20/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_20/bias
k
!dense_20/bias/Read/ReadVariableOpReadVariableOpdense_20/bias*
_output_shapes
:*
dtype0
{
dense_21/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?* 
shared_namedense_21/kernel
t
#dense_21/kernel/Read/ReadVariableOpReadVariableOpdense_21/kernel*
_output_shapes
:	?*
dtype0
r
dense_21/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_21/bias
k
!dense_21/bias/Read/ReadVariableOpReadVariableOpdense_21/bias*
_output_shapes
:*
dtype0
{
dense_22/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?* 
shared_namedense_22/kernel
t
#dense_22/kernel/Read/ReadVariableOpReadVariableOpdense_22/kernel*
_output_shapes
:	?*
dtype0
r
dense_22/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_22/bias
k
!dense_22/bias/Read/ReadVariableOpReadVariableOpdense_22/bias*
_output_shapes
:*
dtype0
{
dense_23/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?* 
shared_namedense_23/kernel
t
#dense_23/kernel/Read/ReadVariableOpReadVariableOpdense_23/kernel*
_output_shapes
:	?*
dtype0
r
dense_23/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_23/bias
k
!dense_23/bias/Read/ReadVariableOpReadVariableOpdense_23/bias*
_output_shapes
:*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
b
total_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_2
[
total_2/Read/ReadVariableOpReadVariableOptotal_2*
_output_shapes
: *
dtype0
b
count_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_2
[
count_2/Read/ReadVariableOpReadVariableOpcount_2*
_output_shapes
: *
dtype0
b
total_3VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_3
[
total_3/Read/ReadVariableOpReadVariableOptotal_3*
_output_shapes
: *
dtype0
b
count_3VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_3
[
count_3/Read/ReadVariableOpReadVariableOpcount_3*
_output_shapes
: *
dtype0
b
total_4VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_4
[
total_4/Read/ReadVariableOpReadVariableOptotal_4*
_output_shapes
: *
dtype0
b
count_4VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_4
[
count_4/Read/ReadVariableOpReadVariableOpcount_4*
_output_shapes
: *
dtype0
b
total_5VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_5
[
total_5/Read/ReadVariableOpReadVariableOptotal_5*
_output_shapes
: *
dtype0
b
count_5VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_5
[
count_5/Read/ReadVariableOpReadVariableOpcount_5*
_output_shapes
: *
dtype0
b
total_6VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_6
[
total_6/Read/ReadVariableOpReadVariableOptotal_6*
_output_shapes
: *
dtype0
b
count_6VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_6
[
count_6/Read/ReadVariableOpReadVariableOpcount_6*
_output_shapes
: *
dtype0
b
total_7VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_7
[
total_7/Read/ReadVariableOpReadVariableOptotal_7*
_output_shapes
: *
dtype0
b
count_7VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_7
[
count_7/Read/ReadVariableOpReadVariableOpcount_7*
_output_shapes
: *
dtype0
b
total_8VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_8
[
total_8/Read/ReadVariableOpReadVariableOptotal_8*
_output_shapes
: *
dtype0
b
count_8VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_8
[
count_8/Read/ReadVariableOpReadVariableOpcount_8*
_output_shapes
: *
dtype0
b
total_9VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_9
[
total_9/Read/ReadVariableOpReadVariableOptotal_9*
_output_shapes
: *
dtype0
b
count_9VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_9
[
count_9/Read/ReadVariableOpReadVariableOpcount_9*
_output_shapes
: *
dtype0
d
total_10VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
total_10
]
total_10/Read/ReadVariableOpReadVariableOptotal_10*
_output_shapes
: *
dtype0
d
count_10VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
count_10
]
count_10/Read/ReadVariableOpReadVariableOpcount_10*
_output_shapes
: *
dtype0
d
total_11VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
total_11
]
total_11/Read/ReadVariableOpReadVariableOptotal_11*
_output_shapes
: *
dtype0
d
count_11VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
count_11
]
count_11/Read/ReadVariableOpReadVariableOpcount_11*
_output_shapes
: *
dtype0
d
total_12VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
total_12
]
total_12/Read/ReadVariableOpReadVariableOptotal_12*
_output_shapes
: *
dtype0
d
count_12VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
count_12
]
count_12/Read/ReadVariableOpReadVariableOpcount_12*
_output_shapes
: *
dtype0
?
Adam/conv3d_3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*'
shared_nameAdam/conv3d_3/kernel/m
?
*Adam/conv3d_3/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv3d_3/kernel/m**
_output_shapes
:d*
dtype0
?
Adam/conv3d_3/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/conv3d_3/bias/m
y
(Adam/conv3d_3/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv3d_3/bias/m*
_output_shapes
:*
dtype0
?
Adam/conv3d_4/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*'
shared_nameAdam/conv3d_4/kernel/m
?
*Adam/conv3d_4/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv3d_4/kernel/m**
_output_shapes
:
*
dtype0
?
Adam/conv3d_4/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*%
shared_nameAdam/conv3d_4/bias/m
y
(Adam/conv3d_4/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv3d_4/bias/m*
_output_shapes
:
*
dtype0
?
Adam/dense_12/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*'
shared_nameAdam/dense_12/kernel/m
?
*Adam/dense_12/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_12/kernel/m*
_output_shapes
:	?*
dtype0
?
Adam/dense_12/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_12/bias/m
y
(Adam/dense_12/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_12/bias/m*
_output_shapes
:*
dtype0
?
Adam/dense_13/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*'
shared_nameAdam/dense_13/kernel/m
?
*Adam/dense_13/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_13/kernel/m*
_output_shapes
:	?*
dtype0
?
Adam/dense_13/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_13/bias/m
y
(Adam/dense_13/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_13/bias/m*
_output_shapes
:*
dtype0
?
Adam/dense_14/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*'
shared_nameAdam/dense_14/kernel/m
?
*Adam/dense_14/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_14/kernel/m*
_output_shapes
:	?*
dtype0
?
Adam/dense_14/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_14/bias/m
y
(Adam/dense_14/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_14/bias/m*
_output_shapes
:*
dtype0
?
Adam/dense_15/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*'
shared_nameAdam/dense_15/kernel/m
?
*Adam/dense_15/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_15/kernel/m*
_output_shapes
:	?*
dtype0
?
Adam/dense_15/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_15/bias/m
y
(Adam/dense_15/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_15/bias/m*
_output_shapes
:*
dtype0
?
Adam/dense_16/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*'
shared_nameAdam/dense_16/kernel/m
?
*Adam/dense_16/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_16/kernel/m*
_output_shapes
:	?*
dtype0
?
Adam/dense_16/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_16/bias/m
y
(Adam/dense_16/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_16/bias/m*
_output_shapes
:*
dtype0
?
Adam/dense_17/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*'
shared_nameAdam/dense_17/kernel/m
?
*Adam/dense_17/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_17/kernel/m*
_output_shapes
:	?*
dtype0
?
Adam/dense_17/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_17/bias/m
y
(Adam/dense_17/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_17/bias/m*
_output_shapes
:*
dtype0
?
Adam/dense_18/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*'
shared_nameAdam/dense_18/kernel/m
?
*Adam/dense_18/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_18/kernel/m*
_output_shapes
:	?*
dtype0
?
Adam/dense_18/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_18/bias/m
y
(Adam/dense_18/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_18/bias/m*
_output_shapes
:*
dtype0
?
Adam/dense_19/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*'
shared_nameAdam/dense_19/kernel/m
?
*Adam/dense_19/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_19/kernel/m*
_output_shapes
:	?*
dtype0
?
Adam/dense_19/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_19/bias/m
y
(Adam/dense_19/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_19/bias/m*
_output_shapes
:*
dtype0
?
Adam/dense_20/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*'
shared_nameAdam/dense_20/kernel/m
?
*Adam/dense_20/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_20/kernel/m*
_output_shapes
:	?*
dtype0
?
Adam/dense_20/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_20/bias/m
y
(Adam/dense_20/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_20/bias/m*
_output_shapes
:*
dtype0
?
Adam/dense_21/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*'
shared_nameAdam/dense_21/kernel/m
?
*Adam/dense_21/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_21/kernel/m*
_output_shapes
:	?*
dtype0
?
Adam/dense_21/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_21/bias/m
y
(Adam/dense_21/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_21/bias/m*
_output_shapes
:*
dtype0
?
Adam/dense_22/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*'
shared_nameAdam/dense_22/kernel/m
?
*Adam/dense_22/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_22/kernel/m*
_output_shapes
:	?*
dtype0
?
Adam/dense_22/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_22/bias/m
y
(Adam/dense_22/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_22/bias/m*
_output_shapes
:*
dtype0
?
Adam/dense_23/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*'
shared_nameAdam/dense_23/kernel/m
?
*Adam/dense_23/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_23/kernel/m*
_output_shapes
:	?*
dtype0
?
Adam/dense_23/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_23/bias/m
y
(Adam/dense_23/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_23/bias/m*
_output_shapes
:*
dtype0
?
Adam/conv3d_3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*'
shared_nameAdam/conv3d_3/kernel/v
?
*Adam/conv3d_3/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv3d_3/kernel/v**
_output_shapes
:d*
dtype0
?
Adam/conv3d_3/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/conv3d_3/bias/v
y
(Adam/conv3d_3/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv3d_3/bias/v*
_output_shapes
:*
dtype0
?
Adam/conv3d_4/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*'
shared_nameAdam/conv3d_4/kernel/v
?
*Adam/conv3d_4/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv3d_4/kernel/v**
_output_shapes
:
*
dtype0
?
Adam/conv3d_4/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*%
shared_nameAdam/conv3d_4/bias/v
y
(Adam/conv3d_4/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv3d_4/bias/v*
_output_shapes
:
*
dtype0
?
Adam/dense_12/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*'
shared_nameAdam/dense_12/kernel/v
?
*Adam/dense_12/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_12/kernel/v*
_output_shapes
:	?*
dtype0
?
Adam/dense_12/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_12/bias/v
y
(Adam/dense_12/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_12/bias/v*
_output_shapes
:*
dtype0
?
Adam/dense_13/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*'
shared_nameAdam/dense_13/kernel/v
?
*Adam/dense_13/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_13/kernel/v*
_output_shapes
:	?*
dtype0
?
Adam/dense_13/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_13/bias/v
y
(Adam/dense_13/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_13/bias/v*
_output_shapes
:*
dtype0
?
Adam/dense_14/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*'
shared_nameAdam/dense_14/kernel/v
?
*Adam/dense_14/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_14/kernel/v*
_output_shapes
:	?*
dtype0
?
Adam/dense_14/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_14/bias/v
y
(Adam/dense_14/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_14/bias/v*
_output_shapes
:*
dtype0
?
Adam/dense_15/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*'
shared_nameAdam/dense_15/kernel/v
?
*Adam/dense_15/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_15/kernel/v*
_output_shapes
:	?*
dtype0
?
Adam/dense_15/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_15/bias/v
y
(Adam/dense_15/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_15/bias/v*
_output_shapes
:*
dtype0
?
Adam/dense_16/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*'
shared_nameAdam/dense_16/kernel/v
?
*Adam/dense_16/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_16/kernel/v*
_output_shapes
:	?*
dtype0
?
Adam/dense_16/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_16/bias/v
y
(Adam/dense_16/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_16/bias/v*
_output_shapes
:*
dtype0
?
Adam/dense_17/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*'
shared_nameAdam/dense_17/kernel/v
?
*Adam/dense_17/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_17/kernel/v*
_output_shapes
:	?*
dtype0
?
Adam/dense_17/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_17/bias/v
y
(Adam/dense_17/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_17/bias/v*
_output_shapes
:*
dtype0
?
Adam/dense_18/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*'
shared_nameAdam/dense_18/kernel/v
?
*Adam/dense_18/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_18/kernel/v*
_output_shapes
:	?*
dtype0
?
Adam/dense_18/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_18/bias/v
y
(Adam/dense_18/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_18/bias/v*
_output_shapes
:*
dtype0
?
Adam/dense_19/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*'
shared_nameAdam/dense_19/kernel/v
?
*Adam/dense_19/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_19/kernel/v*
_output_shapes
:	?*
dtype0
?
Adam/dense_19/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_19/bias/v
y
(Adam/dense_19/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_19/bias/v*
_output_shapes
:*
dtype0
?
Adam/dense_20/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*'
shared_nameAdam/dense_20/kernel/v
?
*Adam/dense_20/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_20/kernel/v*
_output_shapes
:	?*
dtype0
?
Adam/dense_20/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_20/bias/v
y
(Adam/dense_20/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_20/bias/v*
_output_shapes
:*
dtype0
?
Adam/dense_21/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*'
shared_nameAdam/dense_21/kernel/v
?
*Adam/dense_21/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_21/kernel/v*
_output_shapes
:	?*
dtype0
?
Adam/dense_21/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_21/bias/v
y
(Adam/dense_21/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_21/bias/v*
_output_shapes
:*
dtype0
?
Adam/dense_22/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*'
shared_nameAdam/dense_22/kernel/v
?
*Adam/dense_22/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_22/kernel/v*
_output_shapes
:	?*
dtype0
?
Adam/dense_22/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_22/bias/v
y
(Adam/dense_22/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_22/bias/v*
_output_shapes
:*
dtype0
?
Adam/dense_23/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*'
shared_nameAdam/dense_23/kernel/v
?
*Adam/dense_23/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_23/kernel/v*
_output_shapes
:	?*
dtype0
?
Adam/dense_23/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_23/bias/v
y
(Adam/dense_23/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_23/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
??
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*??
value??B?? B??
?
layer-0
layer_with_weights-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer-4
layer-5
layer_with_weights-2
layer-6
layer_with_weights-3
layer-7
	layer_with_weights-4
	layer-8

layer_with_weights-5

layer-9
layer_with_weights-6
layer-10
layer_with_weights-7
layer-11
layer_with_weights-8
layer-12
layer_with_weights-9
layer-13
layer_with_weights-10
layer-14
layer_with_weights-11
layer-15
layer_with_weights-12
layer-16
layer_with_weights-13
layer-17
layer-18
layer-19
layer-20
layer-21
layer-22
layer-23
layer-24
layer-25
layer-26
layer-27
layer-28
layer-29
	optimizer
 loss
!trainable_variables
"	variables
#regularization_losses
$	keras_api
%
signatures
 
h

&kernel
'bias
(trainable_variables
)	variables
*regularization_losses
+	keras_api
R
,trainable_variables
-	variables
.regularization_losses
/	keras_api
h

0kernel
1bias
2trainable_variables
3	variables
4regularization_losses
5	keras_api
R
6trainable_variables
7	variables
8regularization_losses
9	keras_api
R
:trainable_variables
;	variables
<regularization_losses
=	keras_api
h

>kernel
?bias
@trainable_variables
A	variables
Bregularization_losses
C	keras_api
h

Dkernel
Ebias
Ftrainable_variables
G	variables
Hregularization_losses
I	keras_api
h

Jkernel
Kbias
Ltrainable_variables
M	variables
Nregularization_losses
O	keras_api
h

Pkernel
Qbias
Rtrainable_variables
S	variables
Tregularization_losses
U	keras_api
h

Vkernel
Wbias
Xtrainable_variables
Y	variables
Zregularization_losses
[	keras_api
h

\kernel
]bias
^trainable_variables
_	variables
`regularization_losses
a	keras_api
h

bkernel
cbias
dtrainable_variables
e	variables
fregularization_losses
g	keras_api
h

hkernel
ibias
jtrainable_variables
k	variables
lregularization_losses
m	keras_api
h

nkernel
obias
ptrainable_variables
q	variables
rregularization_losses
s	keras_api
h

tkernel
ubias
vtrainable_variables
w	variables
xregularization_losses
y	keras_api
h

zkernel
{bias
|trainable_variables
}	variables
~regularization_losses
	keras_api
n
?kernel
	?bias
?trainable_variables
?	variables
?regularization_losses
?	keras_api
V
?trainable_variables
?	variables
?regularization_losses
?	keras_api
V
?trainable_variables
?	variables
?regularization_losses
?	keras_api
V
?trainable_variables
?	variables
?regularization_losses
?	keras_api
V
?trainable_variables
?	variables
?regularization_losses
?	keras_api
V
?trainable_variables
?	variables
?regularization_losses
?	keras_api
V
?trainable_variables
?	variables
?regularization_losses
?	keras_api
V
?trainable_variables
?	variables
?regularization_losses
?	keras_api
V
?trainable_variables
?	variables
?regularization_losses
?	keras_api
V
?trainable_variables
?	variables
?regularization_losses
?	keras_api
V
?trainable_variables
?	variables
?regularization_losses
?	keras_api
V
?trainable_variables
?	variables
?regularization_losses
?	keras_api
V
?trainable_variables
?	variables
?regularization_losses
?	keras_api
?
	?iter
?beta_1
?beta_2

?decay
?learning_rate&m?'m?0m?1m?>m??m?Dm?Em?Jm?Km?Pm?Qm?Vm?Wm?\m?]m?bm?cm?hm?im?nm?om?tm?um?zm?{m?	?m?	?m?&v?'v?0v?1v?>v??v?Dv?Ev?Jv?Kv?Pv?Qv?Vv?Wv?\v?]v?bv?cv?hv?iv?nv?ov?tv?uv?zv?{v?	?v?	?v?
 
?
&0
'1
02
13
>4
?5
D6
E7
J8
K9
P10
Q11
V12
W13
\14
]15
b16
c17
h18
i19
n20
o21
t22
u23
z24
{25
?26
?27
?
&0
'1
02
13
>4
?5
D6
E7
J8
K9
P10
Q11
V12
W13
\14
]15
b16
c17
h18
i19
n20
o21
t22
u23
z24
{25
?26
?27
 
?
?layer_metrics
!trainable_variables
"	variables
?non_trainable_variables
?metrics
#regularization_losses
 ?layer_regularization_losses
?layers
 
[Y
VARIABLE_VALUEconv3d_3/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv3d_3/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

&0
'1

&0
'1
 
?
?layer_metrics
(trainable_variables
)	variables
?non_trainable_variables
?metrics
*regularization_losses
 ?layer_regularization_losses
?layers
 
 
 
?
?layer_metrics
,trainable_variables
-	variables
?non_trainable_variables
?metrics
.regularization_losses
 ?layer_regularization_losses
?layers
[Y
VARIABLE_VALUEconv3d_4/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv3d_4/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

00
11

00
11
 
?
?layer_metrics
2trainable_variables
3	variables
?non_trainable_variables
?metrics
4regularization_losses
 ?layer_regularization_losses
?layers
 
 
 
?
?layer_metrics
6trainable_variables
7	variables
?non_trainable_variables
?metrics
8regularization_losses
 ?layer_regularization_losses
?layers
 
 
 
?
?layer_metrics
:trainable_variables
;	variables
?non_trainable_variables
?metrics
<regularization_losses
 ?layer_regularization_losses
?layers
[Y
VARIABLE_VALUEdense_12/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_12/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

>0
?1

>0
?1
 
?
?layer_metrics
@trainable_variables
A	variables
?non_trainable_variables
?metrics
Bregularization_losses
 ?layer_regularization_losses
?layers
[Y
VARIABLE_VALUEdense_13/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_13/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE

D0
E1

D0
E1
 
?
?layer_metrics
Ftrainable_variables
G	variables
?non_trainable_variables
?metrics
Hregularization_losses
 ?layer_regularization_losses
?layers
[Y
VARIABLE_VALUEdense_14/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_14/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE

J0
K1

J0
K1
 
?
?layer_metrics
Ltrainable_variables
M	variables
?non_trainable_variables
?metrics
Nregularization_losses
 ?layer_regularization_losses
?layers
[Y
VARIABLE_VALUEdense_15/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_15/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE

P0
Q1

P0
Q1
 
?
?layer_metrics
Rtrainable_variables
S	variables
?non_trainable_variables
?metrics
Tregularization_losses
 ?layer_regularization_losses
?layers
[Y
VARIABLE_VALUEdense_16/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_16/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE

V0
W1

V0
W1
 
?
?layer_metrics
Xtrainable_variables
Y	variables
?non_trainable_variables
?metrics
Zregularization_losses
 ?layer_regularization_losses
?layers
[Y
VARIABLE_VALUEdense_17/kernel6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_17/bias4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUE

\0
]1

\0
]1
 
?
?layer_metrics
^trainable_variables
_	variables
?non_trainable_variables
?metrics
`regularization_losses
 ?layer_regularization_losses
?layers
[Y
VARIABLE_VALUEdense_18/kernel6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_18/bias4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUE

b0
c1

b0
c1
 
?
?layer_metrics
dtrainable_variables
e	variables
?non_trainable_variables
?metrics
fregularization_losses
 ?layer_regularization_losses
?layers
[Y
VARIABLE_VALUEdense_19/kernel6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_19/bias4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUE

h0
i1

h0
i1
 
?
?layer_metrics
jtrainable_variables
k	variables
?non_trainable_variables
?metrics
lregularization_losses
 ?layer_regularization_losses
?layers
\Z
VARIABLE_VALUEdense_20/kernel7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEdense_20/bias5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUE

n0
o1

n0
o1
 
?
?layer_metrics
ptrainable_variables
q	variables
?non_trainable_variables
?metrics
rregularization_losses
 ?layer_regularization_losses
?layers
\Z
VARIABLE_VALUEdense_21/kernel7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEdense_21/bias5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUE

t0
u1

t0
u1
 
?
?layer_metrics
vtrainable_variables
w	variables
?non_trainable_variables
?metrics
xregularization_losses
 ?layer_regularization_losses
?layers
\Z
VARIABLE_VALUEdense_22/kernel7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEdense_22/bias5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUE

z0
{1

z0
{1
 
?
?layer_metrics
|trainable_variables
}	variables
?non_trainable_variables
?metrics
~regularization_losses
 ?layer_regularization_losses
?layers
\Z
VARIABLE_VALUEdense_23/kernel7layer_with_weights-13/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEdense_23/bias5layer_with_weights-13/bias/.ATTRIBUTES/VARIABLE_VALUE

?0
?1

?0
?1
 
?
?layer_metrics
?trainable_variables
?	variables
?non_trainable_variables
?metrics
?regularization_losses
 ?layer_regularization_losses
?layers
 
 
 
?
?layer_metrics
?trainable_variables
?	variables
?non_trainable_variables
?metrics
?regularization_losses
 ?layer_regularization_losses
?layers
 
 
 
?
?layer_metrics
?trainable_variables
?	variables
?non_trainable_variables
?metrics
?regularization_losses
 ?layer_regularization_losses
?layers
 
 
 
?
?layer_metrics
?trainable_variables
?	variables
?non_trainable_variables
?metrics
?regularization_losses
 ?layer_regularization_losses
?layers
 
 
 
?
?layer_metrics
?trainable_variables
?	variables
?non_trainable_variables
?metrics
?regularization_losses
 ?layer_regularization_losses
?layers
 
 
 
?
?layer_metrics
?trainable_variables
?	variables
?non_trainable_variables
?metrics
?regularization_losses
 ?layer_regularization_losses
?layers
 
 
 
?
?layer_metrics
?trainable_variables
?	variables
?non_trainable_variables
?metrics
?regularization_losses
 ?layer_regularization_losses
?layers
 
 
 
?
?layer_metrics
?trainable_variables
?	variables
?non_trainable_variables
?metrics
?regularization_losses
 ?layer_regularization_losses
?layers
 
 
 
?
?layer_metrics
?trainable_variables
?	variables
?non_trainable_variables
?metrics
?regularization_losses
 ?layer_regularization_losses
?layers
 
 
 
?
?layer_metrics
?trainable_variables
?	variables
?non_trainable_variables
?metrics
?regularization_losses
 ?layer_regularization_losses
?layers
 
 
 
?
?layer_metrics
?trainable_variables
?	variables
?non_trainable_variables
?metrics
?regularization_losses
 ?layer_regularization_losses
?layers
 
 
 
?
?layer_metrics
?trainable_variables
?	variables
?non_trainable_variables
?metrics
?regularization_losses
 ?layer_regularization_losses
?layers
 
 
 
?
?layer_metrics
?trainable_variables
?	variables
?non_trainable_variables
?metrics
?regularization_losses
 ?layer_regularization_losses
?layers
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
 
 
k
?0
?1
?2
?3
?4
?5
?6
?7
?8
?9
?10
?11
?12
 
?
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
29
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
8

?total

?count
?	variables
?	keras_api
8

?total

?count
?	variables
?	keras_api
8

?total

?count
?	variables
?	keras_api
8

?total

?count
?	variables
?	keras_api
8

?total

?count
?	variables
?	keras_api
8

?total

?count
?	variables
?	keras_api
8

?total

?count
?	variables
?	keras_api
8

?total

?count
?	variables
?	keras_api
8

?total

?count
?	variables
?	keras_api
8

?total

?count
?	variables
?	keras_api
8

?total

?count
?	variables
?	keras_api
8

?total

?count
?	variables
?	keras_api
8

?total

?count
?	variables
?	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

?0
?1

?	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE

?0
?1

?	variables
QO
VARIABLE_VALUEtotal_24keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_24keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUE

?0
?1

?	variables
QO
VARIABLE_VALUEtotal_34keras_api/metrics/3/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_34keras_api/metrics/3/count/.ATTRIBUTES/VARIABLE_VALUE

?0
?1

?	variables
QO
VARIABLE_VALUEtotal_44keras_api/metrics/4/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_44keras_api/metrics/4/count/.ATTRIBUTES/VARIABLE_VALUE

?0
?1

?	variables
QO
VARIABLE_VALUEtotal_54keras_api/metrics/5/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_54keras_api/metrics/5/count/.ATTRIBUTES/VARIABLE_VALUE

?0
?1

?	variables
QO
VARIABLE_VALUEtotal_64keras_api/metrics/6/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_64keras_api/metrics/6/count/.ATTRIBUTES/VARIABLE_VALUE

?0
?1

?	variables
QO
VARIABLE_VALUEtotal_74keras_api/metrics/7/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_74keras_api/metrics/7/count/.ATTRIBUTES/VARIABLE_VALUE

?0
?1

?	variables
QO
VARIABLE_VALUEtotal_84keras_api/metrics/8/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_84keras_api/metrics/8/count/.ATTRIBUTES/VARIABLE_VALUE

?0
?1

?	variables
QO
VARIABLE_VALUEtotal_94keras_api/metrics/9/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_94keras_api/metrics/9/count/.ATTRIBUTES/VARIABLE_VALUE

?0
?1

?	variables
SQ
VARIABLE_VALUEtotal_105keras_api/metrics/10/total/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUEcount_105keras_api/metrics/10/count/.ATTRIBUTES/VARIABLE_VALUE

?0
?1

?	variables
SQ
VARIABLE_VALUEtotal_115keras_api/metrics/11/total/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUEcount_115keras_api/metrics/11/count/.ATTRIBUTES/VARIABLE_VALUE

?0
?1

?	variables
SQ
VARIABLE_VALUEtotal_125keras_api/metrics/12/total/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUEcount_125keras_api/metrics/12/count/.ATTRIBUTES/VARIABLE_VALUE

?0
?1

?	variables
~|
VARIABLE_VALUEAdam/conv3d_3/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv3d_3/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv3d_4/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv3d_4/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_12/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_12/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_13/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_13/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_14/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_14/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_15/kernel/mRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_15/bias/mPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_16/kernel/mRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_16/bias/mPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_17/kernel/mRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_17/bias/mPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_18/kernel/mRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_18/bias/mPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_19/kernel/mRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_19/bias/mPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/dense_20/kernel/mSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense_20/bias/mQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/dense_21/kernel/mSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense_21/bias/mQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/dense_22/kernel/mSlayer_with_weights-12/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense_22/bias/mQlayer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/dense_23/kernel/mSlayer_with_weights-13/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense_23/bias/mQlayer_with_weights-13/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv3d_3/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv3d_3/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv3d_4/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv3d_4/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_12/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_12/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_13/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_13/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_14/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_14/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_15/kernel/vRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_15/bias/vPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_16/kernel/vRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_16/bias/vPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_17/kernel/vRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_17/bias/vPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_18/kernel/vRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_18/bias/vPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_19/kernel/vRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_19/bias/vPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/dense_20/kernel/vSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense_20/bias/vQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/dense_21/kernel/vSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense_21/bias/vQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/dense_22/kernel/vSlayer_with_weights-12/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense_22/bias/vQlayer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/dense_23/kernel/vSlayer_with_weights-13/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense_23/bias/vQlayer_with_weights-13/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?
serving_default_input_2Placeholder*3
_output_shapes!
:?????????ppd*
dtype0*(
shape:?????????ppd
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_2conv3d_3/kernelconv3d_3/biasconv3d_4/kernelconv3d_4/biasdense_23/kerneldense_23/biasdense_22/kerneldense_22/biasdense_21/kerneldense_21/biasdense_20/kerneldense_20/biasdense_19/kerneldense_19/biasdense_18/kerneldense_18/biasdense_17/kerneldense_17/biasdense_16/kerneldense_16/biasdense_15/kerneldense_15/biasdense_14/kerneldense_14/biasdense_13/kerneldense_13/biasdense_12/kerneldense_12/bias*(
Tin!
2*
Tout
2*
_collective_manager_ids
 *?
_output_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????*>
_read_only_resource_inputs 
	
*-
config_proto

CPU

GPU 2J 8? *+
f&R$
"__inference_signature_wrapper_9416
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?%
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename#conv3d_3/kernel/Read/ReadVariableOp!conv3d_3/bias/Read/ReadVariableOp#conv3d_4/kernel/Read/ReadVariableOp!conv3d_4/bias/Read/ReadVariableOp#dense_12/kernel/Read/ReadVariableOp!dense_12/bias/Read/ReadVariableOp#dense_13/kernel/Read/ReadVariableOp!dense_13/bias/Read/ReadVariableOp#dense_14/kernel/Read/ReadVariableOp!dense_14/bias/Read/ReadVariableOp#dense_15/kernel/Read/ReadVariableOp!dense_15/bias/Read/ReadVariableOp#dense_16/kernel/Read/ReadVariableOp!dense_16/bias/Read/ReadVariableOp#dense_17/kernel/Read/ReadVariableOp!dense_17/bias/Read/ReadVariableOp#dense_18/kernel/Read/ReadVariableOp!dense_18/bias/Read/ReadVariableOp#dense_19/kernel/Read/ReadVariableOp!dense_19/bias/Read/ReadVariableOp#dense_20/kernel/Read/ReadVariableOp!dense_20/bias/Read/ReadVariableOp#dense_21/kernel/Read/ReadVariableOp!dense_21/bias/Read/ReadVariableOp#dense_22/kernel/Read/ReadVariableOp!dense_22/bias/Read/ReadVariableOp#dense_23/kernel/Read/ReadVariableOp!dense_23/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOptotal_2/Read/ReadVariableOpcount_2/Read/ReadVariableOptotal_3/Read/ReadVariableOpcount_3/Read/ReadVariableOptotal_4/Read/ReadVariableOpcount_4/Read/ReadVariableOptotal_5/Read/ReadVariableOpcount_5/Read/ReadVariableOptotal_6/Read/ReadVariableOpcount_6/Read/ReadVariableOptotal_7/Read/ReadVariableOpcount_7/Read/ReadVariableOptotal_8/Read/ReadVariableOpcount_8/Read/ReadVariableOptotal_9/Read/ReadVariableOpcount_9/Read/ReadVariableOptotal_10/Read/ReadVariableOpcount_10/Read/ReadVariableOptotal_11/Read/ReadVariableOpcount_11/Read/ReadVariableOptotal_12/Read/ReadVariableOpcount_12/Read/ReadVariableOp*Adam/conv3d_3/kernel/m/Read/ReadVariableOp(Adam/conv3d_3/bias/m/Read/ReadVariableOp*Adam/conv3d_4/kernel/m/Read/ReadVariableOp(Adam/conv3d_4/bias/m/Read/ReadVariableOp*Adam/dense_12/kernel/m/Read/ReadVariableOp(Adam/dense_12/bias/m/Read/ReadVariableOp*Adam/dense_13/kernel/m/Read/ReadVariableOp(Adam/dense_13/bias/m/Read/ReadVariableOp*Adam/dense_14/kernel/m/Read/ReadVariableOp(Adam/dense_14/bias/m/Read/ReadVariableOp*Adam/dense_15/kernel/m/Read/ReadVariableOp(Adam/dense_15/bias/m/Read/ReadVariableOp*Adam/dense_16/kernel/m/Read/ReadVariableOp(Adam/dense_16/bias/m/Read/ReadVariableOp*Adam/dense_17/kernel/m/Read/ReadVariableOp(Adam/dense_17/bias/m/Read/ReadVariableOp*Adam/dense_18/kernel/m/Read/ReadVariableOp(Adam/dense_18/bias/m/Read/ReadVariableOp*Adam/dense_19/kernel/m/Read/ReadVariableOp(Adam/dense_19/bias/m/Read/ReadVariableOp*Adam/dense_20/kernel/m/Read/ReadVariableOp(Adam/dense_20/bias/m/Read/ReadVariableOp*Adam/dense_21/kernel/m/Read/ReadVariableOp(Adam/dense_21/bias/m/Read/ReadVariableOp*Adam/dense_22/kernel/m/Read/ReadVariableOp(Adam/dense_22/bias/m/Read/ReadVariableOp*Adam/dense_23/kernel/m/Read/ReadVariableOp(Adam/dense_23/bias/m/Read/ReadVariableOp*Adam/conv3d_3/kernel/v/Read/ReadVariableOp(Adam/conv3d_3/bias/v/Read/ReadVariableOp*Adam/conv3d_4/kernel/v/Read/ReadVariableOp(Adam/conv3d_4/bias/v/Read/ReadVariableOp*Adam/dense_12/kernel/v/Read/ReadVariableOp(Adam/dense_12/bias/v/Read/ReadVariableOp*Adam/dense_13/kernel/v/Read/ReadVariableOp(Adam/dense_13/bias/v/Read/ReadVariableOp*Adam/dense_14/kernel/v/Read/ReadVariableOp(Adam/dense_14/bias/v/Read/ReadVariableOp*Adam/dense_15/kernel/v/Read/ReadVariableOp(Adam/dense_15/bias/v/Read/ReadVariableOp*Adam/dense_16/kernel/v/Read/ReadVariableOp(Adam/dense_16/bias/v/Read/ReadVariableOp*Adam/dense_17/kernel/v/Read/ReadVariableOp(Adam/dense_17/bias/v/Read/ReadVariableOp*Adam/dense_18/kernel/v/Read/ReadVariableOp(Adam/dense_18/bias/v/Read/ReadVariableOp*Adam/dense_19/kernel/v/Read/ReadVariableOp(Adam/dense_19/bias/v/Read/ReadVariableOp*Adam/dense_20/kernel/v/Read/ReadVariableOp(Adam/dense_20/bias/v/Read/ReadVariableOp*Adam/dense_21/kernel/v/Read/ReadVariableOp(Adam/dense_21/bias/v/Read/ReadVariableOp*Adam/dense_22/kernel/v/Read/ReadVariableOp(Adam/dense_22/bias/v/Read/ReadVariableOp*Adam/dense_23/kernel/v/Read/ReadVariableOp(Adam/dense_23/bias/v/Read/ReadVariableOpConst*?
Tiny
w2u	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *'
f"R 
__inference__traced_save_10666
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv3d_3/kernelconv3d_3/biasconv3d_4/kernelconv3d_4/biasdense_12/kerneldense_12/biasdense_13/kerneldense_13/biasdense_14/kerneldense_14/biasdense_15/kerneldense_15/biasdense_16/kerneldense_16/biasdense_17/kerneldense_17/biasdense_18/kerneldense_18/biasdense_19/kerneldense_19/biasdense_20/kerneldense_20/biasdense_21/kerneldense_21/biasdense_22/kerneldense_22/biasdense_23/kerneldense_23/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcounttotal_1count_1total_2count_2total_3count_3total_4count_4total_5count_5total_6count_6total_7count_7total_8count_8total_9count_9total_10count_10total_11count_11total_12count_12Adam/conv3d_3/kernel/mAdam/conv3d_3/bias/mAdam/conv3d_4/kernel/mAdam/conv3d_4/bias/mAdam/dense_12/kernel/mAdam/dense_12/bias/mAdam/dense_13/kernel/mAdam/dense_13/bias/mAdam/dense_14/kernel/mAdam/dense_14/bias/mAdam/dense_15/kernel/mAdam/dense_15/bias/mAdam/dense_16/kernel/mAdam/dense_16/bias/mAdam/dense_17/kernel/mAdam/dense_17/bias/mAdam/dense_18/kernel/mAdam/dense_18/bias/mAdam/dense_19/kernel/mAdam/dense_19/bias/mAdam/dense_20/kernel/mAdam/dense_20/bias/mAdam/dense_21/kernel/mAdam/dense_21/bias/mAdam/dense_22/kernel/mAdam/dense_22/bias/mAdam/dense_23/kernel/mAdam/dense_23/bias/mAdam/conv3d_3/kernel/vAdam/conv3d_3/bias/vAdam/conv3d_4/kernel/vAdam/conv3d_4/bias/vAdam/dense_12/kernel/vAdam/dense_12/bias/vAdam/dense_13/kernel/vAdam/dense_13/bias/vAdam/dense_14/kernel/vAdam/dense_14/bias/vAdam/dense_15/kernel/vAdam/dense_15/bias/vAdam/dense_16/kernel/vAdam/dense_16/bias/vAdam/dense_17/kernel/vAdam/dense_17/bias/vAdam/dense_18/kernel/vAdam/dense_18/bias/vAdam/dense_19/kernel/vAdam/dense_19/bias/vAdam/dense_20/kernel/vAdam/dense_20/bias/vAdam/dense_21/kernel/vAdam/dense_21/bias/vAdam/dense_22/kernel/vAdam/dense_22/bias/vAdam/dense_23/kernel/vAdam/dense_23/bias/v*
Tinx
v2t*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? **
f%R#
!__inference__traced_restore_11021??
?
a
E__inference_flatten_19_layer_call_and_return_conditional_losses_10216

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
Constg
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:?????????2	
Reshaped
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
|
'__inference_dense_15_layer_call_fn_9995

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_15_layer_call_and_return_conditional_losses_85792
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
`
D__inference_flatten_20_layer_call_and_return_conditional_losses_8752

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
Constg
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:?????????2	
Reshaped
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
a
E__inference_flatten_23_layer_call_and_return_conditional_losses_10260

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
Constg
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:?????????2	
Reshaped
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
B__inference_dense_15_layer_call_and_return_conditional_losses_8579

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
a
E__inference_flatten_24_layer_call_and_return_conditional_losses_10271

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
Constg
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:?????????2	
Reshaped
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
}
(__inference_dense_22_layer_call_fn_10135

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_22_layer_call_and_return_conditional_losses_83902
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
??
?,
__inference__traced_save_10666
file_prefix.
*savev2_conv3d_3_kernel_read_readvariableop,
(savev2_conv3d_3_bias_read_readvariableop.
*savev2_conv3d_4_kernel_read_readvariableop,
(savev2_conv3d_4_bias_read_readvariableop.
*savev2_dense_12_kernel_read_readvariableop,
(savev2_dense_12_bias_read_readvariableop.
*savev2_dense_13_kernel_read_readvariableop,
(savev2_dense_13_bias_read_readvariableop.
*savev2_dense_14_kernel_read_readvariableop,
(savev2_dense_14_bias_read_readvariableop.
*savev2_dense_15_kernel_read_readvariableop,
(savev2_dense_15_bias_read_readvariableop.
*savev2_dense_16_kernel_read_readvariableop,
(savev2_dense_16_bias_read_readvariableop.
*savev2_dense_17_kernel_read_readvariableop,
(savev2_dense_17_bias_read_readvariableop.
*savev2_dense_18_kernel_read_readvariableop,
(savev2_dense_18_bias_read_readvariableop.
*savev2_dense_19_kernel_read_readvariableop,
(savev2_dense_19_bias_read_readvariableop.
*savev2_dense_20_kernel_read_readvariableop,
(savev2_dense_20_bias_read_readvariableop.
*savev2_dense_21_kernel_read_readvariableop,
(savev2_dense_21_bias_read_readvariableop.
*savev2_dense_22_kernel_read_readvariableop,
(savev2_dense_22_bias_read_readvariableop.
*savev2_dense_23_kernel_read_readvariableop,
(savev2_dense_23_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop&
"savev2_total_2_read_readvariableop&
"savev2_count_2_read_readvariableop&
"savev2_total_3_read_readvariableop&
"savev2_count_3_read_readvariableop&
"savev2_total_4_read_readvariableop&
"savev2_count_4_read_readvariableop&
"savev2_total_5_read_readvariableop&
"savev2_count_5_read_readvariableop&
"savev2_total_6_read_readvariableop&
"savev2_count_6_read_readvariableop&
"savev2_total_7_read_readvariableop&
"savev2_count_7_read_readvariableop&
"savev2_total_8_read_readvariableop&
"savev2_count_8_read_readvariableop&
"savev2_total_9_read_readvariableop&
"savev2_count_9_read_readvariableop'
#savev2_total_10_read_readvariableop'
#savev2_count_10_read_readvariableop'
#savev2_total_11_read_readvariableop'
#savev2_count_11_read_readvariableop'
#savev2_total_12_read_readvariableop'
#savev2_count_12_read_readvariableop5
1savev2_adam_conv3d_3_kernel_m_read_readvariableop3
/savev2_adam_conv3d_3_bias_m_read_readvariableop5
1savev2_adam_conv3d_4_kernel_m_read_readvariableop3
/savev2_adam_conv3d_4_bias_m_read_readvariableop5
1savev2_adam_dense_12_kernel_m_read_readvariableop3
/savev2_adam_dense_12_bias_m_read_readvariableop5
1savev2_adam_dense_13_kernel_m_read_readvariableop3
/savev2_adam_dense_13_bias_m_read_readvariableop5
1savev2_adam_dense_14_kernel_m_read_readvariableop3
/savev2_adam_dense_14_bias_m_read_readvariableop5
1savev2_adam_dense_15_kernel_m_read_readvariableop3
/savev2_adam_dense_15_bias_m_read_readvariableop5
1savev2_adam_dense_16_kernel_m_read_readvariableop3
/savev2_adam_dense_16_bias_m_read_readvariableop5
1savev2_adam_dense_17_kernel_m_read_readvariableop3
/savev2_adam_dense_17_bias_m_read_readvariableop5
1savev2_adam_dense_18_kernel_m_read_readvariableop3
/savev2_adam_dense_18_bias_m_read_readvariableop5
1savev2_adam_dense_19_kernel_m_read_readvariableop3
/savev2_adam_dense_19_bias_m_read_readvariableop5
1savev2_adam_dense_20_kernel_m_read_readvariableop3
/savev2_adam_dense_20_bias_m_read_readvariableop5
1savev2_adam_dense_21_kernel_m_read_readvariableop3
/savev2_adam_dense_21_bias_m_read_readvariableop5
1savev2_adam_dense_22_kernel_m_read_readvariableop3
/savev2_adam_dense_22_bias_m_read_readvariableop5
1savev2_adam_dense_23_kernel_m_read_readvariableop3
/savev2_adam_dense_23_bias_m_read_readvariableop5
1savev2_adam_conv3d_3_kernel_v_read_readvariableop3
/savev2_adam_conv3d_3_bias_v_read_readvariableop5
1savev2_adam_conv3d_4_kernel_v_read_readvariableop3
/savev2_adam_conv3d_4_bias_v_read_readvariableop5
1savev2_adam_dense_12_kernel_v_read_readvariableop3
/savev2_adam_dense_12_bias_v_read_readvariableop5
1savev2_adam_dense_13_kernel_v_read_readvariableop3
/savev2_adam_dense_13_bias_v_read_readvariableop5
1savev2_adam_dense_14_kernel_v_read_readvariableop3
/savev2_adam_dense_14_bias_v_read_readvariableop5
1savev2_adam_dense_15_kernel_v_read_readvariableop3
/savev2_adam_dense_15_bias_v_read_readvariableop5
1savev2_adam_dense_16_kernel_v_read_readvariableop3
/savev2_adam_dense_16_bias_v_read_readvariableop5
1savev2_adam_dense_17_kernel_v_read_readvariableop3
/savev2_adam_dense_17_bias_v_read_readvariableop5
1savev2_adam_dense_18_kernel_v_read_readvariableop3
/savev2_adam_dense_18_bias_v_read_readvariableop5
1savev2_adam_dense_19_kernel_v_read_readvariableop3
/savev2_adam_dense_19_bias_v_read_readvariableop5
1savev2_adam_dense_20_kernel_v_read_readvariableop3
/savev2_adam_dense_20_bias_v_read_readvariableop5
1savev2_adam_dense_21_kernel_v_read_readvariableop3
/savev2_adam_dense_21_bias_v_read_readvariableop5
1savev2_adam_dense_22_kernel_v_read_readvariableop3
/savev2_adam_dense_22_bias_v_read_readvariableop5
1savev2_adam_dense_23_kernel_v_read_readvariableop3
/savev2_adam_dense_23_bias_v_read_readvariableop
savev2_const

identity_1??MergeV2Checkpoints?
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename?>
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:t*
dtype0*?=
value?=B?=tB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-13/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-13/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/3/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/3/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/4/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/4/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/5/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/5/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/6/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/6/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/7/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/7/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/8/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/8/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/9/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/9/count/.ATTRIBUTES/VARIABLE_VALUEB5keras_api/metrics/10/total/.ATTRIBUTES/VARIABLE_VALUEB5keras_api/metrics/10/count/.ATTRIBUTES/VARIABLE_VALUEB5keras_api/metrics/11/total/.ATTRIBUTES/VARIABLE_VALUEB5keras_api/metrics/11/count/.ATTRIBUTES/VARIABLE_VALUEB5keras_api/metrics/12/total/.ATTRIBUTES/VARIABLE_VALUEB5keras_api/metrics/12/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-12/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-13/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-13/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-12/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-13/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-13/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:t*
dtype0*?
value?B?tB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?*
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0*savev2_conv3d_3_kernel_read_readvariableop(savev2_conv3d_3_bias_read_readvariableop*savev2_conv3d_4_kernel_read_readvariableop(savev2_conv3d_4_bias_read_readvariableop*savev2_dense_12_kernel_read_readvariableop(savev2_dense_12_bias_read_readvariableop*savev2_dense_13_kernel_read_readvariableop(savev2_dense_13_bias_read_readvariableop*savev2_dense_14_kernel_read_readvariableop(savev2_dense_14_bias_read_readvariableop*savev2_dense_15_kernel_read_readvariableop(savev2_dense_15_bias_read_readvariableop*savev2_dense_16_kernel_read_readvariableop(savev2_dense_16_bias_read_readvariableop*savev2_dense_17_kernel_read_readvariableop(savev2_dense_17_bias_read_readvariableop*savev2_dense_18_kernel_read_readvariableop(savev2_dense_18_bias_read_readvariableop*savev2_dense_19_kernel_read_readvariableop(savev2_dense_19_bias_read_readvariableop*savev2_dense_20_kernel_read_readvariableop(savev2_dense_20_bias_read_readvariableop*savev2_dense_21_kernel_read_readvariableop(savev2_dense_21_bias_read_readvariableop*savev2_dense_22_kernel_read_readvariableop(savev2_dense_22_bias_read_readvariableop*savev2_dense_23_kernel_read_readvariableop(savev2_dense_23_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop"savev2_total_2_read_readvariableop"savev2_count_2_read_readvariableop"savev2_total_3_read_readvariableop"savev2_count_3_read_readvariableop"savev2_total_4_read_readvariableop"savev2_count_4_read_readvariableop"savev2_total_5_read_readvariableop"savev2_count_5_read_readvariableop"savev2_total_6_read_readvariableop"savev2_count_6_read_readvariableop"savev2_total_7_read_readvariableop"savev2_count_7_read_readvariableop"savev2_total_8_read_readvariableop"savev2_count_8_read_readvariableop"savev2_total_9_read_readvariableop"savev2_count_9_read_readvariableop#savev2_total_10_read_readvariableop#savev2_count_10_read_readvariableop#savev2_total_11_read_readvariableop#savev2_count_11_read_readvariableop#savev2_total_12_read_readvariableop#savev2_count_12_read_readvariableop1savev2_adam_conv3d_3_kernel_m_read_readvariableop/savev2_adam_conv3d_3_bias_m_read_readvariableop1savev2_adam_conv3d_4_kernel_m_read_readvariableop/savev2_adam_conv3d_4_bias_m_read_readvariableop1savev2_adam_dense_12_kernel_m_read_readvariableop/savev2_adam_dense_12_bias_m_read_readvariableop1savev2_adam_dense_13_kernel_m_read_readvariableop/savev2_adam_dense_13_bias_m_read_readvariableop1savev2_adam_dense_14_kernel_m_read_readvariableop/savev2_adam_dense_14_bias_m_read_readvariableop1savev2_adam_dense_15_kernel_m_read_readvariableop/savev2_adam_dense_15_bias_m_read_readvariableop1savev2_adam_dense_16_kernel_m_read_readvariableop/savev2_adam_dense_16_bias_m_read_readvariableop1savev2_adam_dense_17_kernel_m_read_readvariableop/savev2_adam_dense_17_bias_m_read_readvariableop1savev2_adam_dense_18_kernel_m_read_readvariableop/savev2_adam_dense_18_bias_m_read_readvariableop1savev2_adam_dense_19_kernel_m_read_readvariableop/savev2_adam_dense_19_bias_m_read_readvariableop1savev2_adam_dense_20_kernel_m_read_readvariableop/savev2_adam_dense_20_bias_m_read_readvariableop1savev2_adam_dense_21_kernel_m_read_readvariableop/savev2_adam_dense_21_bias_m_read_readvariableop1savev2_adam_dense_22_kernel_m_read_readvariableop/savev2_adam_dense_22_bias_m_read_readvariableop1savev2_adam_dense_23_kernel_m_read_readvariableop/savev2_adam_dense_23_bias_m_read_readvariableop1savev2_adam_conv3d_3_kernel_v_read_readvariableop/savev2_adam_conv3d_3_bias_v_read_readvariableop1savev2_adam_conv3d_4_kernel_v_read_readvariableop/savev2_adam_conv3d_4_bias_v_read_readvariableop1savev2_adam_dense_12_kernel_v_read_readvariableop/savev2_adam_dense_12_bias_v_read_readvariableop1savev2_adam_dense_13_kernel_v_read_readvariableop/savev2_adam_dense_13_bias_v_read_readvariableop1savev2_adam_dense_14_kernel_v_read_readvariableop/savev2_adam_dense_14_bias_v_read_readvariableop1savev2_adam_dense_15_kernel_v_read_readvariableop/savev2_adam_dense_15_bias_v_read_readvariableop1savev2_adam_dense_16_kernel_v_read_readvariableop/savev2_adam_dense_16_bias_v_read_readvariableop1savev2_adam_dense_17_kernel_v_read_readvariableop/savev2_adam_dense_17_bias_v_read_readvariableop1savev2_adam_dense_18_kernel_v_read_readvariableop/savev2_adam_dense_18_bias_v_read_readvariableop1savev2_adam_dense_19_kernel_v_read_readvariableop/savev2_adam_dense_19_bias_v_read_readvariableop1savev2_adam_dense_20_kernel_v_read_readvariableop/savev2_adam_dense_20_bias_v_read_readvariableop1savev2_adam_dense_21_kernel_v_read_readvariableop/savev2_adam_dense_21_bias_v_read_readvariableop1savev2_adam_dense_22_kernel_v_read_readvariableop/savev2_adam_dense_22_bias_v_read_readvariableop1savev2_adam_dense_23_kernel_v_read_readvariableop/savev2_adam_dense_23_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *?
dtypesx
v2t	2
SaveV2?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*?
_input_shapes?
?: :d::
:
:	?::	?::	?::	?::	?::	?::	?::	?::	?::	?::	?::	?:: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : :d::
:
:	?::	?::	?::	?::	?::	?::	?::	?::	?::	?::	?::	?::d::
:
:	?::	?::	?::	?::	?::	?::	?::	?::	?::	?::	?::	?:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:0,
*
_output_shapes
:d: 

_output_shapes
::0,
*
_output_shapes
:
: 

_output_shapes
:
:%!

_output_shapes
:	?: 

_output_shapes
::%!

_output_shapes
:	?: 

_output_shapes
::%	!

_output_shapes
:	?: 


_output_shapes
::%!

_output_shapes
:	?: 

_output_shapes
::%!

_output_shapes
:	?: 

_output_shapes
::%!

_output_shapes
:	?: 

_output_shapes
::%!

_output_shapes
:	?: 

_output_shapes
::%!

_output_shapes
:	?: 

_output_shapes
::%!

_output_shapes
:	?: 

_output_shapes
::%!

_output_shapes
:	?: 

_output_shapes
::%!

_output_shapes
:	?: 

_output_shapes
::%!

_output_shapes
:	?: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: : 

_output_shapes
: :!

_output_shapes
: :"

_output_shapes
: :#

_output_shapes
: :$

_output_shapes
: :%

_output_shapes
: :&

_output_shapes
: :'

_output_shapes
: :(

_output_shapes
: :)

_output_shapes
: :*

_output_shapes
: :+

_output_shapes
: :,

_output_shapes
: :-

_output_shapes
: :.

_output_shapes
: :/

_output_shapes
: :0

_output_shapes
: :1

_output_shapes
: :2

_output_shapes
: :3

_output_shapes
: :4

_output_shapes
: :5

_output_shapes
: :6

_output_shapes
: :7

_output_shapes
: :8

_output_shapes
: :9

_output_shapes
: ::

_output_shapes
: :;

_output_shapes
: :0<,
*
_output_shapes
:d: =

_output_shapes
::0>,
*
_output_shapes
:
: ?

_output_shapes
:
:%@!

_output_shapes
:	?: A

_output_shapes
::%B!

_output_shapes
:	?: C

_output_shapes
::%D!

_output_shapes
:	?: E

_output_shapes
::%F!

_output_shapes
:	?: G

_output_shapes
::%H!

_output_shapes
:	?: I

_output_shapes
::%J!

_output_shapes
:	?: K

_output_shapes
::%L!

_output_shapes
:	?: M

_output_shapes
::%N!

_output_shapes
:	?: O

_output_shapes
::%P!

_output_shapes
:	?: Q

_output_shapes
::%R!

_output_shapes
:	?: S

_output_shapes
::%T!

_output_shapes
:	?: U

_output_shapes
::%V!

_output_shapes
:	?: W

_output_shapes
::0X,
*
_output_shapes
:d: Y

_output_shapes
::0Z,
*
_output_shapes
:
: [

_output_shapes
:
:%\!

_output_shapes
:	?: ]

_output_shapes
::%^!

_output_shapes
:	?: _

_output_shapes
::%`!

_output_shapes
:	?: a

_output_shapes
::%b!

_output_shapes
:	?: c

_output_shapes
::%d!

_output_shapes
:	?: e

_output_shapes
::%f!

_output_shapes
:	?: g

_output_shapes
::%h!

_output_shapes
:	?: i

_output_shapes
::%j!

_output_shapes
:	?: k

_output_shapes
::%l!

_output_shapes
:	?: m

_output_shapes
::%n!

_output_shapes
:	?: o

_output_shapes
::%p!

_output_shapes
:	?: q

_output_shapes
::%r!

_output_shapes
:	?: s

_output_shapes
::t

_output_shapes
: 
?
F
*__inference_flatten_24_layer_call_fn_10276

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_flatten_24_layer_call_and_return_conditional_losses_86962
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
F
*__inference_flatten_20_layer_call_fn_10232

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_flatten_20_layer_call_and_return_conditional_losses_87522
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
B__inference_dense_13_layer_call_and_return_conditional_losses_9946

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
}
(__inference_dense_18_layer_call_fn_10055

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_18_layer_call_and_return_conditional_losses_84982
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?	
?
B__inference_dense_19_layer_call_and_return_conditional_losses_8471

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
??
?

A__inference_model_1_layer_call_and_return_conditional_losses_8856
input_2
conv3d_3_8304
conv3d_3_8306
conv3d_4_8332
conv3d_4_8334
dense_23_8374
dense_23_8376
dense_22_8401
dense_22_8403
dense_21_8428
dense_21_8430
dense_20_8455
dense_20_8457
dense_19_8482
dense_19_8484
dense_18_8509
dense_18_8511
dense_17_8536
dense_17_8538
dense_16_8563
dense_16_8565
dense_15_8590
dense_15_8592
dense_14_8617
dense_14_8619
dense_13_8644
dense_13_8646
dense_12_8671
dense_12_8673
identity

identity_1

identity_2

identity_3

identity_4

identity_5

identity_6

identity_7

identity_8

identity_9
identity_10
identity_11?? conv3d_3/StatefulPartitionedCall? conv3d_4/StatefulPartitionedCall? dense_12/StatefulPartitionedCall? dense_13/StatefulPartitionedCall? dense_14/StatefulPartitionedCall? dense_15/StatefulPartitionedCall? dense_16/StatefulPartitionedCall? dense_17/StatefulPartitionedCall? dense_18/StatefulPartitionedCall? dense_19/StatefulPartitionedCall? dense_20/StatefulPartitionedCall? dense_21/StatefulPartitionedCall? dense_22/StatefulPartitionedCall? dense_23/StatefulPartitionedCall?
 conv3d_3/StatefulPartitionedCallStatefulPartitionedCallinput_2conv3d_3_8304conv3d_3_8306*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????66*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_conv3d_3_layer_call_and_return_conditional_losses_82932"
 conv3d_3/StatefulPartitionedCall?
max_pooling3d_2/PartitionedCallPartitionedCall)conv3d_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_max_pooling3d_2_layer_call_and_return_conditional_losses_82602!
max_pooling3d_2/PartitionedCall?
 conv3d_4/StatefulPartitionedCallStatefulPartitionedCall(max_pooling3d_2/PartitionedCall:output:0conv3d_4_8332conv3d_4_8334*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_conv3d_4_layer_call_and_return_conditional_losses_83212"
 conv3d_4/StatefulPartitionedCall?
max_pooling3d_3/PartitionedCallPartitionedCall)conv3d_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_max_pooling3d_3_layer_call_and_return_conditional_losses_82722!
max_pooling3d_3/PartitionedCall?
flatten_13/PartitionedCallPartitionedCall(max_pooling3d_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_flatten_13_layer_call_and_return_conditional_losses_83442
flatten_13/PartitionedCall?
 dense_23/StatefulPartitionedCallStatefulPartitionedCall#flatten_13/PartitionedCall:output:0dense_23_8374dense_23_8376*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_23_layer_call_and_return_conditional_losses_83632"
 dense_23/StatefulPartitionedCall?
 dense_22/StatefulPartitionedCallStatefulPartitionedCall#flatten_13/PartitionedCall:output:0dense_22_8401dense_22_8403*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_22_layer_call_and_return_conditional_losses_83902"
 dense_22/StatefulPartitionedCall?
 dense_21/StatefulPartitionedCallStatefulPartitionedCall#flatten_13/PartitionedCall:output:0dense_21_8428dense_21_8430*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_21_layer_call_and_return_conditional_losses_84172"
 dense_21/StatefulPartitionedCall?
 dense_20/StatefulPartitionedCallStatefulPartitionedCall#flatten_13/PartitionedCall:output:0dense_20_8455dense_20_8457*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_20_layer_call_and_return_conditional_losses_84442"
 dense_20/StatefulPartitionedCall?
 dense_19/StatefulPartitionedCallStatefulPartitionedCall#flatten_13/PartitionedCall:output:0dense_19_8482dense_19_8484*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_19_layer_call_and_return_conditional_losses_84712"
 dense_19/StatefulPartitionedCall?
 dense_18/StatefulPartitionedCallStatefulPartitionedCall#flatten_13/PartitionedCall:output:0dense_18_8509dense_18_8511*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_18_layer_call_and_return_conditional_losses_84982"
 dense_18/StatefulPartitionedCall?
 dense_17/StatefulPartitionedCallStatefulPartitionedCall#flatten_13/PartitionedCall:output:0dense_17_8536dense_17_8538*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_17_layer_call_and_return_conditional_losses_85252"
 dense_17/StatefulPartitionedCall?
 dense_16/StatefulPartitionedCallStatefulPartitionedCall#flatten_13/PartitionedCall:output:0dense_16_8563dense_16_8565*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_16_layer_call_and_return_conditional_losses_85522"
 dense_16/StatefulPartitionedCall?
 dense_15/StatefulPartitionedCallStatefulPartitionedCall#flatten_13/PartitionedCall:output:0dense_15_8590dense_15_8592*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_15_layer_call_and_return_conditional_losses_85792"
 dense_15/StatefulPartitionedCall?
 dense_14/StatefulPartitionedCallStatefulPartitionedCall#flatten_13/PartitionedCall:output:0dense_14_8617dense_14_8619*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_14_layer_call_and_return_conditional_losses_86062"
 dense_14/StatefulPartitionedCall?
 dense_13/StatefulPartitionedCallStatefulPartitionedCall#flatten_13/PartitionedCall:output:0dense_13_8644dense_13_8646*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_13_layer_call_and_return_conditional_losses_86332"
 dense_13/StatefulPartitionedCall?
 dense_12/StatefulPartitionedCallStatefulPartitionedCall#flatten_13/PartitionedCall:output:0dense_12_8671dense_12_8673*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_12_layer_call_and_return_conditional_losses_86602"
 dense_12/StatefulPartitionedCall?
flatten_25/PartitionedCallPartitionedCall)dense_23/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_flatten_25_layer_call_and_return_conditional_losses_86822
flatten_25/PartitionedCall?
flatten_24/PartitionedCallPartitionedCall)dense_22/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_flatten_24_layer_call_and_return_conditional_losses_86962
flatten_24/PartitionedCall?
flatten_23/PartitionedCallPartitionedCall)dense_21/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_flatten_23_layer_call_and_return_conditional_losses_87102
flatten_23/PartitionedCall?
flatten_22/PartitionedCallPartitionedCall)dense_20/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_flatten_22_layer_call_and_return_conditional_losses_87242
flatten_22/PartitionedCall?
flatten_21/PartitionedCallPartitionedCall)dense_19/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_flatten_21_layer_call_and_return_conditional_losses_87382
flatten_21/PartitionedCall?
flatten_20/PartitionedCallPartitionedCall)dense_18/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_flatten_20_layer_call_and_return_conditional_losses_87522
flatten_20/PartitionedCall?
flatten_19/PartitionedCallPartitionedCall)dense_17/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_flatten_19_layer_call_and_return_conditional_losses_87662
flatten_19/PartitionedCall?
flatten_18/PartitionedCallPartitionedCall)dense_16/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_flatten_18_layer_call_and_return_conditional_losses_87802
flatten_18/PartitionedCall?
flatten_17/PartitionedCallPartitionedCall)dense_15/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_flatten_17_layer_call_and_return_conditional_losses_87942
flatten_17/PartitionedCall?
flatten_16/PartitionedCallPartitionedCall)dense_14/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_flatten_16_layer_call_and_return_conditional_losses_88082
flatten_16/PartitionedCall?
flatten_15/PartitionedCallPartitionedCall)dense_13/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_flatten_15_layer_call_and_return_conditional_losses_88222
flatten_15/PartitionedCall?
flatten_14/PartitionedCallPartitionedCall)dense_12/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_flatten_14_layer_call_and_return_conditional_losses_88362
flatten_14/PartitionedCall?
IdentityIdentity#flatten_14/PartitionedCall:output:0!^conv3d_3/StatefulPartitionedCall!^conv3d_4/StatefulPartitionedCall!^dense_12/StatefulPartitionedCall!^dense_13/StatefulPartitionedCall!^dense_14/StatefulPartitionedCall!^dense_15/StatefulPartitionedCall!^dense_16/StatefulPartitionedCall!^dense_17/StatefulPartitionedCall!^dense_18/StatefulPartitionedCall!^dense_19/StatefulPartitionedCall!^dense_20/StatefulPartitionedCall!^dense_21/StatefulPartitionedCall!^dense_22/StatefulPartitionedCall!^dense_23/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity?

Identity_1Identity#flatten_15/PartitionedCall:output:0!^conv3d_3/StatefulPartitionedCall!^conv3d_4/StatefulPartitionedCall!^dense_12/StatefulPartitionedCall!^dense_13/StatefulPartitionedCall!^dense_14/StatefulPartitionedCall!^dense_15/StatefulPartitionedCall!^dense_16/StatefulPartitionedCall!^dense_17/StatefulPartitionedCall!^dense_18/StatefulPartitionedCall!^dense_19/StatefulPartitionedCall!^dense_20/StatefulPartitionedCall!^dense_21/StatefulPartitionedCall!^dense_22/StatefulPartitionedCall!^dense_23/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_1?

Identity_2Identity#flatten_16/PartitionedCall:output:0!^conv3d_3/StatefulPartitionedCall!^conv3d_4/StatefulPartitionedCall!^dense_12/StatefulPartitionedCall!^dense_13/StatefulPartitionedCall!^dense_14/StatefulPartitionedCall!^dense_15/StatefulPartitionedCall!^dense_16/StatefulPartitionedCall!^dense_17/StatefulPartitionedCall!^dense_18/StatefulPartitionedCall!^dense_19/StatefulPartitionedCall!^dense_20/StatefulPartitionedCall!^dense_21/StatefulPartitionedCall!^dense_22/StatefulPartitionedCall!^dense_23/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_2?

Identity_3Identity#flatten_17/PartitionedCall:output:0!^conv3d_3/StatefulPartitionedCall!^conv3d_4/StatefulPartitionedCall!^dense_12/StatefulPartitionedCall!^dense_13/StatefulPartitionedCall!^dense_14/StatefulPartitionedCall!^dense_15/StatefulPartitionedCall!^dense_16/StatefulPartitionedCall!^dense_17/StatefulPartitionedCall!^dense_18/StatefulPartitionedCall!^dense_19/StatefulPartitionedCall!^dense_20/StatefulPartitionedCall!^dense_21/StatefulPartitionedCall!^dense_22/StatefulPartitionedCall!^dense_23/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_3?

Identity_4Identity#flatten_18/PartitionedCall:output:0!^conv3d_3/StatefulPartitionedCall!^conv3d_4/StatefulPartitionedCall!^dense_12/StatefulPartitionedCall!^dense_13/StatefulPartitionedCall!^dense_14/StatefulPartitionedCall!^dense_15/StatefulPartitionedCall!^dense_16/StatefulPartitionedCall!^dense_17/StatefulPartitionedCall!^dense_18/StatefulPartitionedCall!^dense_19/StatefulPartitionedCall!^dense_20/StatefulPartitionedCall!^dense_21/StatefulPartitionedCall!^dense_22/StatefulPartitionedCall!^dense_23/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_4?

Identity_5Identity#flatten_19/PartitionedCall:output:0!^conv3d_3/StatefulPartitionedCall!^conv3d_4/StatefulPartitionedCall!^dense_12/StatefulPartitionedCall!^dense_13/StatefulPartitionedCall!^dense_14/StatefulPartitionedCall!^dense_15/StatefulPartitionedCall!^dense_16/StatefulPartitionedCall!^dense_17/StatefulPartitionedCall!^dense_18/StatefulPartitionedCall!^dense_19/StatefulPartitionedCall!^dense_20/StatefulPartitionedCall!^dense_21/StatefulPartitionedCall!^dense_22/StatefulPartitionedCall!^dense_23/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_5?

Identity_6Identity#flatten_20/PartitionedCall:output:0!^conv3d_3/StatefulPartitionedCall!^conv3d_4/StatefulPartitionedCall!^dense_12/StatefulPartitionedCall!^dense_13/StatefulPartitionedCall!^dense_14/StatefulPartitionedCall!^dense_15/StatefulPartitionedCall!^dense_16/StatefulPartitionedCall!^dense_17/StatefulPartitionedCall!^dense_18/StatefulPartitionedCall!^dense_19/StatefulPartitionedCall!^dense_20/StatefulPartitionedCall!^dense_21/StatefulPartitionedCall!^dense_22/StatefulPartitionedCall!^dense_23/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_6?

Identity_7Identity#flatten_21/PartitionedCall:output:0!^conv3d_3/StatefulPartitionedCall!^conv3d_4/StatefulPartitionedCall!^dense_12/StatefulPartitionedCall!^dense_13/StatefulPartitionedCall!^dense_14/StatefulPartitionedCall!^dense_15/StatefulPartitionedCall!^dense_16/StatefulPartitionedCall!^dense_17/StatefulPartitionedCall!^dense_18/StatefulPartitionedCall!^dense_19/StatefulPartitionedCall!^dense_20/StatefulPartitionedCall!^dense_21/StatefulPartitionedCall!^dense_22/StatefulPartitionedCall!^dense_23/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_7?

Identity_8Identity#flatten_22/PartitionedCall:output:0!^conv3d_3/StatefulPartitionedCall!^conv3d_4/StatefulPartitionedCall!^dense_12/StatefulPartitionedCall!^dense_13/StatefulPartitionedCall!^dense_14/StatefulPartitionedCall!^dense_15/StatefulPartitionedCall!^dense_16/StatefulPartitionedCall!^dense_17/StatefulPartitionedCall!^dense_18/StatefulPartitionedCall!^dense_19/StatefulPartitionedCall!^dense_20/StatefulPartitionedCall!^dense_21/StatefulPartitionedCall!^dense_22/StatefulPartitionedCall!^dense_23/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_8?

Identity_9Identity#flatten_23/PartitionedCall:output:0!^conv3d_3/StatefulPartitionedCall!^conv3d_4/StatefulPartitionedCall!^dense_12/StatefulPartitionedCall!^dense_13/StatefulPartitionedCall!^dense_14/StatefulPartitionedCall!^dense_15/StatefulPartitionedCall!^dense_16/StatefulPartitionedCall!^dense_17/StatefulPartitionedCall!^dense_18/StatefulPartitionedCall!^dense_19/StatefulPartitionedCall!^dense_20/StatefulPartitionedCall!^dense_21/StatefulPartitionedCall!^dense_22/StatefulPartitionedCall!^dense_23/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_9?
Identity_10Identity#flatten_24/PartitionedCall:output:0!^conv3d_3/StatefulPartitionedCall!^conv3d_4/StatefulPartitionedCall!^dense_12/StatefulPartitionedCall!^dense_13/StatefulPartitionedCall!^dense_14/StatefulPartitionedCall!^dense_15/StatefulPartitionedCall!^dense_16/StatefulPartitionedCall!^dense_17/StatefulPartitionedCall!^dense_18/StatefulPartitionedCall!^dense_19/StatefulPartitionedCall!^dense_20/StatefulPartitionedCall!^dense_21/StatefulPartitionedCall!^dense_22/StatefulPartitionedCall!^dense_23/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2
Identity_10?
Identity_11Identity#flatten_25/PartitionedCall:output:0!^conv3d_3/StatefulPartitionedCall!^conv3d_4/StatefulPartitionedCall!^dense_12/StatefulPartitionedCall!^dense_13/StatefulPartitionedCall!^dense_14/StatefulPartitionedCall!^dense_15/StatefulPartitionedCall!^dense_16/StatefulPartitionedCall!^dense_17/StatefulPartitionedCall!^dense_18/StatefulPartitionedCall!^dense_19/StatefulPartitionedCall!^dense_20/StatefulPartitionedCall!^dense_21/StatefulPartitionedCall!^dense_22/StatefulPartitionedCall!^dense_23/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2
Identity_11"
identityIdentity:output:0"!

identity_1Identity_1:output:0"#
identity_10Identity_10:output:0"#
identity_11Identity_11:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"!

identity_6Identity_6:output:0"!

identity_7Identity_7:output:0"!

identity_8Identity_8:output:0"!

identity_9Identity_9:output:0*?
_input_shapes?
?:?????????ppd::::::::::::::::::::::::::::2D
 conv3d_3/StatefulPartitionedCall conv3d_3/StatefulPartitionedCall2D
 conv3d_4/StatefulPartitionedCall conv3d_4/StatefulPartitionedCall2D
 dense_12/StatefulPartitionedCall dense_12/StatefulPartitionedCall2D
 dense_13/StatefulPartitionedCall dense_13/StatefulPartitionedCall2D
 dense_14/StatefulPartitionedCall dense_14/StatefulPartitionedCall2D
 dense_15/StatefulPartitionedCall dense_15/StatefulPartitionedCall2D
 dense_16/StatefulPartitionedCall dense_16/StatefulPartitionedCall2D
 dense_17/StatefulPartitionedCall dense_17/StatefulPartitionedCall2D
 dense_18/StatefulPartitionedCall dense_18/StatefulPartitionedCall2D
 dense_19/StatefulPartitionedCall dense_19/StatefulPartitionedCall2D
 dense_20/StatefulPartitionedCall dense_20/StatefulPartitionedCall2D
 dense_21/StatefulPartitionedCall dense_21/StatefulPartitionedCall2D
 dense_22/StatefulPartitionedCall dense_22/StatefulPartitionedCall2D
 dense_23/StatefulPartitionedCall dense_23/StatefulPartitionedCall:\ X
3
_output_shapes!
:?????????ppd
!
_user_specified_name	input_2
?
a
E__inference_flatten_15_layer_call_and_return_conditional_losses_10172

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
Constg
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:?????????2	
Reshaped
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
B__inference_dense_17_layer_call_and_return_conditional_losses_8525

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
F
*__inference_flatten_23_layer_call_fn_10265

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_flatten_23_layer_call_and_return_conditional_losses_87102
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
}
(__inference_dense_16_layer_call_fn_10015

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_16_layer_call_and_return_conditional_losses_85522
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?

?
B__inference_conv3d_4_layer_call_and_return_conditional_losses_9895

inputs"
conv3d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv3D/ReadVariableOp?
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:
*
dtype02
Conv3D/ReadVariableOp?
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????
*
paddingVALID*
strides	
2
Conv3D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????
2	
BiasAddd
ReluReluBiasAdd:output:0*
T0*3
_output_shapes!
:?????????
2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*
T0*3
_output_shapes!
:?????????
2

Identity"
identityIdentity:output:0*:
_input_shapes)
':?????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:[ W
3
_output_shapes!
:?????????
 
_user_specified_nameinputs
?
F
*__inference_flatten_16_layer_call_fn_10188

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_flatten_16_layer_call_and_return_conditional_losses_88082
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
}
(__inference_dense_23_layer_call_fn_10155

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_23_layer_call_and_return_conditional_losses_83632
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
? 
?
&__inference_model_1_layer_call_fn_9140
input_2
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26
identity

identity_1

identity_2

identity_3

identity_4

identity_5

identity_6

identity_7

identity_8

identity_9
identity_10
identity_11??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26*(
Tin!
2*
Tout
2*
_collective_manager_ids
 *?
_output_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????*>
_read_only_resource_inputs 
	
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_model_1_layer_call_and_return_conditional_losses_90592
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity?

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_1?

Identity_2Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_2?

Identity_3Identity StatefulPartitionedCall:output:3^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_3?

Identity_4Identity StatefulPartitionedCall:output:4^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_4?

Identity_5Identity StatefulPartitionedCall:output:5^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_5?

Identity_6Identity StatefulPartitionedCall:output:6^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_6?

Identity_7Identity StatefulPartitionedCall:output:7^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_7?

Identity_8Identity StatefulPartitionedCall:output:8^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_8?

Identity_9Identity StatefulPartitionedCall:output:9^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_9?
Identity_10Identity!StatefulPartitionedCall:output:10^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2
Identity_10?
Identity_11Identity!StatefulPartitionedCall:output:11^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2
Identity_11"
identityIdentity:output:0"!

identity_1Identity_1:output:0"#
identity_10Identity_10:output:0"#
identity_11Identity_11:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"!

identity_6Identity_6:output:0"!

identity_7Identity_7:output:0"!

identity_8Identity_8:output:0"!

identity_9Identity_9:output:0*?
_input_shapes?
?:?????????ppd::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:\ X
3
_output_shapes!
:?????????ppd
!
_user_specified_name	input_2
?
a
E__inference_flatten_16_layer_call_and_return_conditional_losses_10183

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
Constg
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:?????????2	
Reshaped
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
C__inference_dense_21_layer_call_and_return_conditional_losses_10106

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?	
?
B__inference_dense_14_layer_call_and_return_conditional_losses_9966

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
F
*__inference_flatten_19_layer_call_fn_10221

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_flatten_19_layer_call_and_return_conditional_losses_87662
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
B__inference_dense_20_layer_call_and_return_conditional_losses_8444

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?	
?
B__inference_dense_22_layer_call_and_return_conditional_losses_8390

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
??
?
A__inference_model_1_layer_call_and_return_conditional_losses_9557

inputs+
'conv3d_3_conv3d_readvariableop_resource,
(conv3d_3_biasadd_readvariableop_resource+
'conv3d_4_conv3d_readvariableop_resource,
(conv3d_4_biasadd_readvariableop_resource+
'dense_23_matmul_readvariableop_resource,
(dense_23_biasadd_readvariableop_resource+
'dense_22_matmul_readvariableop_resource,
(dense_22_biasadd_readvariableop_resource+
'dense_21_matmul_readvariableop_resource,
(dense_21_biasadd_readvariableop_resource+
'dense_20_matmul_readvariableop_resource,
(dense_20_biasadd_readvariableop_resource+
'dense_19_matmul_readvariableop_resource,
(dense_19_biasadd_readvariableop_resource+
'dense_18_matmul_readvariableop_resource,
(dense_18_biasadd_readvariableop_resource+
'dense_17_matmul_readvariableop_resource,
(dense_17_biasadd_readvariableop_resource+
'dense_16_matmul_readvariableop_resource,
(dense_16_biasadd_readvariableop_resource+
'dense_15_matmul_readvariableop_resource,
(dense_15_biasadd_readvariableop_resource+
'dense_14_matmul_readvariableop_resource,
(dense_14_biasadd_readvariableop_resource+
'dense_13_matmul_readvariableop_resource,
(dense_13_biasadd_readvariableop_resource+
'dense_12_matmul_readvariableop_resource,
(dense_12_biasadd_readvariableop_resource
identity

identity_1

identity_2

identity_3

identity_4

identity_5

identity_6

identity_7

identity_8

identity_9
identity_10
identity_11??conv3d_3/BiasAdd/ReadVariableOp?conv3d_3/Conv3D/ReadVariableOp?conv3d_4/BiasAdd/ReadVariableOp?conv3d_4/Conv3D/ReadVariableOp?dense_12/BiasAdd/ReadVariableOp?dense_12/MatMul/ReadVariableOp?dense_13/BiasAdd/ReadVariableOp?dense_13/MatMul/ReadVariableOp?dense_14/BiasAdd/ReadVariableOp?dense_14/MatMul/ReadVariableOp?dense_15/BiasAdd/ReadVariableOp?dense_15/MatMul/ReadVariableOp?dense_16/BiasAdd/ReadVariableOp?dense_16/MatMul/ReadVariableOp?dense_17/BiasAdd/ReadVariableOp?dense_17/MatMul/ReadVariableOp?dense_18/BiasAdd/ReadVariableOp?dense_18/MatMul/ReadVariableOp?dense_19/BiasAdd/ReadVariableOp?dense_19/MatMul/ReadVariableOp?dense_20/BiasAdd/ReadVariableOp?dense_20/MatMul/ReadVariableOp?dense_21/BiasAdd/ReadVariableOp?dense_21/MatMul/ReadVariableOp?dense_22/BiasAdd/ReadVariableOp?dense_22/MatMul/ReadVariableOp?dense_23/BiasAdd/ReadVariableOp?dense_23/MatMul/ReadVariableOp?
conv3d_3/Conv3D/ReadVariableOpReadVariableOp'conv3d_3_conv3d_readvariableop_resource**
_output_shapes
:d*
dtype02 
conv3d_3/Conv3D/ReadVariableOp?
conv3d_3/Conv3DConv3Dinputs&conv3d_3/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????66*
paddingVALID*
strides	
2
conv3d_3/Conv3D?
conv3d_3/BiasAdd/ReadVariableOpReadVariableOp(conv3d_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
conv3d_3/BiasAdd/ReadVariableOp?
conv3d_3/BiasAddBiasAddconv3d_3/Conv3D:output:0'conv3d_3/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????662
conv3d_3/BiasAdd
conv3d_3/ReluReluconv3d_3/BiasAdd:output:0*
T0*3
_output_shapes!
:?????????662
conv3d_3/Relu?
max_pooling3d_2/MaxPool3D	MaxPool3Dconv3d_3/Relu:activations:0*
T0*3
_output_shapes!
:?????????*
ksize	
*
paddingVALID*
strides	
2
max_pooling3d_2/MaxPool3D?
conv3d_4/Conv3D/ReadVariableOpReadVariableOp'conv3d_4_conv3d_readvariableop_resource**
_output_shapes
:
*
dtype02 
conv3d_4/Conv3D/ReadVariableOp?
conv3d_4/Conv3DConv3D"max_pooling3d_2/MaxPool3D:output:0&conv3d_4/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????
*
paddingVALID*
strides	
2
conv3d_4/Conv3D?
conv3d_4/BiasAdd/ReadVariableOpReadVariableOp(conv3d_4_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02!
conv3d_4/BiasAdd/ReadVariableOp?
conv3d_4/BiasAddBiasAddconv3d_4/Conv3D:output:0'conv3d_4/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????
2
conv3d_4/BiasAdd
conv3d_4/ReluReluconv3d_4/BiasAdd:output:0*
T0*3
_output_shapes!
:?????????
2
conv3d_4/Relu?
max_pooling3d_3/MaxPool3D	MaxPool3Dconv3d_4/Relu:activations:0*
T0*3
_output_shapes!
:?????????
*
ksize	
*
paddingVALID*
strides	
2
max_pooling3d_3/MaxPool3Du
flatten_13/ConstConst*
_output_shapes
:*
dtype0*
valueB"????h  2
flatten_13/Const?
flatten_13/ReshapeReshape"max_pooling3d_3/MaxPool3D:output:0flatten_13/Const:output:0*
T0*(
_output_shapes
:??????????2
flatten_13/Reshape?
dense_23/MatMul/ReadVariableOpReadVariableOp'dense_23_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02 
dense_23/MatMul/ReadVariableOp?
dense_23/MatMulMatMulflatten_13/Reshape:output:0&dense_23/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_23/MatMul?
dense_23/BiasAdd/ReadVariableOpReadVariableOp(dense_23_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_23/BiasAdd/ReadVariableOp?
dense_23/BiasAddBiasAdddense_23/MatMul:product:0'dense_23/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_23/BiasAdd|
dense_23/SigmoidSigmoiddense_23/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_23/Sigmoid?
dense_22/MatMul/ReadVariableOpReadVariableOp'dense_22_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02 
dense_22/MatMul/ReadVariableOp?
dense_22/MatMulMatMulflatten_13/Reshape:output:0&dense_22/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_22/MatMul?
dense_22/BiasAdd/ReadVariableOpReadVariableOp(dense_22_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_22/BiasAdd/ReadVariableOp?
dense_22/BiasAddBiasAdddense_22/MatMul:product:0'dense_22/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_22/BiasAdds
dense_22/ReluReludense_22/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_22/Relu?
dense_21/MatMul/ReadVariableOpReadVariableOp'dense_21_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02 
dense_21/MatMul/ReadVariableOp?
dense_21/MatMulMatMulflatten_13/Reshape:output:0&dense_21/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_21/MatMul?
dense_21/BiasAdd/ReadVariableOpReadVariableOp(dense_21_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_21/BiasAdd/ReadVariableOp?
dense_21/BiasAddBiasAdddense_21/MatMul:product:0'dense_21/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_21/BiasAdds
dense_21/ReluReludense_21/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_21/Relu?
dense_20/MatMul/ReadVariableOpReadVariableOp'dense_20_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02 
dense_20/MatMul/ReadVariableOp?
dense_20/MatMulMatMulflatten_13/Reshape:output:0&dense_20/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_20/MatMul?
dense_20/BiasAdd/ReadVariableOpReadVariableOp(dense_20_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_20/BiasAdd/ReadVariableOp?
dense_20/BiasAddBiasAdddense_20/MatMul:product:0'dense_20/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_20/BiasAdds
dense_20/ReluReludense_20/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_20/Relu?
dense_19/MatMul/ReadVariableOpReadVariableOp'dense_19_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02 
dense_19/MatMul/ReadVariableOp?
dense_19/MatMulMatMulflatten_13/Reshape:output:0&dense_19/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_19/MatMul?
dense_19/BiasAdd/ReadVariableOpReadVariableOp(dense_19_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_19/BiasAdd/ReadVariableOp?
dense_19/BiasAddBiasAdddense_19/MatMul:product:0'dense_19/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_19/BiasAdds
dense_19/ReluReludense_19/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_19/Relu?
dense_18/MatMul/ReadVariableOpReadVariableOp'dense_18_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02 
dense_18/MatMul/ReadVariableOp?
dense_18/MatMulMatMulflatten_13/Reshape:output:0&dense_18/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_18/MatMul?
dense_18/BiasAdd/ReadVariableOpReadVariableOp(dense_18_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_18/BiasAdd/ReadVariableOp?
dense_18/BiasAddBiasAdddense_18/MatMul:product:0'dense_18/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_18/BiasAdds
dense_18/ReluReludense_18/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_18/Relu?
dense_17/MatMul/ReadVariableOpReadVariableOp'dense_17_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02 
dense_17/MatMul/ReadVariableOp?
dense_17/MatMulMatMulflatten_13/Reshape:output:0&dense_17/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_17/MatMul?
dense_17/BiasAdd/ReadVariableOpReadVariableOp(dense_17_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_17/BiasAdd/ReadVariableOp?
dense_17/BiasAddBiasAdddense_17/MatMul:product:0'dense_17/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_17/BiasAdds
dense_17/ReluReludense_17/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_17/Relu?
dense_16/MatMul/ReadVariableOpReadVariableOp'dense_16_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02 
dense_16/MatMul/ReadVariableOp?
dense_16/MatMulMatMulflatten_13/Reshape:output:0&dense_16/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_16/MatMul?
dense_16/BiasAdd/ReadVariableOpReadVariableOp(dense_16_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_16/BiasAdd/ReadVariableOp?
dense_16/BiasAddBiasAdddense_16/MatMul:product:0'dense_16/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_16/BiasAdds
dense_16/ReluReludense_16/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_16/Relu?
dense_15/MatMul/ReadVariableOpReadVariableOp'dense_15_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02 
dense_15/MatMul/ReadVariableOp?
dense_15/MatMulMatMulflatten_13/Reshape:output:0&dense_15/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_15/MatMul?
dense_15/BiasAdd/ReadVariableOpReadVariableOp(dense_15_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_15/BiasAdd/ReadVariableOp?
dense_15/BiasAddBiasAdddense_15/MatMul:product:0'dense_15/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_15/BiasAdds
dense_15/ReluReludense_15/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_15/Relu?
dense_14/MatMul/ReadVariableOpReadVariableOp'dense_14_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02 
dense_14/MatMul/ReadVariableOp?
dense_14/MatMulMatMulflatten_13/Reshape:output:0&dense_14/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_14/MatMul?
dense_14/BiasAdd/ReadVariableOpReadVariableOp(dense_14_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_14/BiasAdd/ReadVariableOp?
dense_14/BiasAddBiasAdddense_14/MatMul:product:0'dense_14/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_14/BiasAdds
dense_14/ReluReludense_14/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_14/Relu?
dense_13/MatMul/ReadVariableOpReadVariableOp'dense_13_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02 
dense_13/MatMul/ReadVariableOp?
dense_13/MatMulMatMulflatten_13/Reshape:output:0&dense_13/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_13/MatMul?
dense_13/BiasAdd/ReadVariableOpReadVariableOp(dense_13_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_13/BiasAdd/ReadVariableOp?
dense_13/BiasAddBiasAdddense_13/MatMul:product:0'dense_13/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_13/BiasAdds
dense_13/ReluReludense_13/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_13/Relu?
dense_12/MatMul/ReadVariableOpReadVariableOp'dense_12_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02 
dense_12/MatMul/ReadVariableOp?
dense_12/MatMulMatMulflatten_13/Reshape:output:0&dense_12/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_12/MatMul?
dense_12/BiasAdd/ReadVariableOpReadVariableOp(dense_12_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_12/BiasAdd/ReadVariableOp?
dense_12/BiasAddBiasAdddense_12/MatMul:product:0'dense_12/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_12/BiasAdds
dense_12/ReluReludense_12/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_12/Reluu
flatten_25/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
flatten_25/Const?
flatten_25/ReshapeReshapedense_23/Sigmoid:y:0flatten_25/Const:output:0*
T0*'
_output_shapes
:?????????2
flatten_25/Reshapeu
flatten_24/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
flatten_24/Const?
flatten_24/ReshapeReshapedense_22/Relu:activations:0flatten_24/Const:output:0*
T0*'
_output_shapes
:?????????2
flatten_24/Reshapeu
flatten_23/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
flatten_23/Const?
flatten_23/ReshapeReshapedense_21/Relu:activations:0flatten_23/Const:output:0*
T0*'
_output_shapes
:?????????2
flatten_23/Reshapeu
flatten_22/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
flatten_22/Const?
flatten_22/ReshapeReshapedense_20/Relu:activations:0flatten_22/Const:output:0*
T0*'
_output_shapes
:?????????2
flatten_22/Reshapeu
flatten_21/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
flatten_21/Const?
flatten_21/ReshapeReshapedense_19/Relu:activations:0flatten_21/Const:output:0*
T0*'
_output_shapes
:?????????2
flatten_21/Reshapeu
flatten_20/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
flatten_20/Const?
flatten_20/ReshapeReshapedense_18/Relu:activations:0flatten_20/Const:output:0*
T0*'
_output_shapes
:?????????2
flatten_20/Reshapeu
flatten_19/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
flatten_19/Const?
flatten_19/ReshapeReshapedense_17/Relu:activations:0flatten_19/Const:output:0*
T0*'
_output_shapes
:?????????2
flatten_19/Reshapeu
flatten_18/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
flatten_18/Const?
flatten_18/ReshapeReshapedense_16/Relu:activations:0flatten_18/Const:output:0*
T0*'
_output_shapes
:?????????2
flatten_18/Reshapeu
flatten_17/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
flatten_17/Const?
flatten_17/ReshapeReshapedense_15/Relu:activations:0flatten_17/Const:output:0*
T0*'
_output_shapes
:?????????2
flatten_17/Reshapeu
flatten_16/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
flatten_16/Const?
flatten_16/ReshapeReshapedense_14/Relu:activations:0flatten_16/Const:output:0*
T0*'
_output_shapes
:?????????2
flatten_16/Reshapeu
flatten_15/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
flatten_15/Const?
flatten_15/ReshapeReshapedense_13/Relu:activations:0flatten_15/Const:output:0*
T0*'
_output_shapes
:?????????2
flatten_15/Reshapeu
flatten_14/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
flatten_14/Const?
flatten_14/ReshapeReshapedense_12/Relu:activations:0flatten_14/Const:output:0*
T0*'
_output_shapes
:?????????2
flatten_14/Reshape?
IdentityIdentityflatten_14/Reshape:output:0 ^conv3d_3/BiasAdd/ReadVariableOp^conv3d_3/Conv3D/ReadVariableOp ^conv3d_4/BiasAdd/ReadVariableOp^conv3d_4/Conv3D/ReadVariableOp ^dense_12/BiasAdd/ReadVariableOp^dense_12/MatMul/ReadVariableOp ^dense_13/BiasAdd/ReadVariableOp^dense_13/MatMul/ReadVariableOp ^dense_14/BiasAdd/ReadVariableOp^dense_14/MatMul/ReadVariableOp ^dense_15/BiasAdd/ReadVariableOp^dense_15/MatMul/ReadVariableOp ^dense_16/BiasAdd/ReadVariableOp^dense_16/MatMul/ReadVariableOp ^dense_17/BiasAdd/ReadVariableOp^dense_17/MatMul/ReadVariableOp ^dense_18/BiasAdd/ReadVariableOp^dense_18/MatMul/ReadVariableOp ^dense_19/BiasAdd/ReadVariableOp^dense_19/MatMul/ReadVariableOp ^dense_20/BiasAdd/ReadVariableOp^dense_20/MatMul/ReadVariableOp ^dense_21/BiasAdd/ReadVariableOp^dense_21/MatMul/ReadVariableOp ^dense_22/BiasAdd/ReadVariableOp^dense_22/MatMul/ReadVariableOp ^dense_23/BiasAdd/ReadVariableOp^dense_23/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity?

Identity_1Identityflatten_15/Reshape:output:0 ^conv3d_3/BiasAdd/ReadVariableOp^conv3d_3/Conv3D/ReadVariableOp ^conv3d_4/BiasAdd/ReadVariableOp^conv3d_4/Conv3D/ReadVariableOp ^dense_12/BiasAdd/ReadVariableOp^dense_12/MatMul/ReadVariableOp ^dense_13/BiasAdd/ReadVariableOp^dense_13/MatMul/ReadVariableOp ^dense_14/BiasAdd/ReadVariableOp^dense_14/MatMul/ReadVariableOp ^dense_15/BiasAdd/ReadVariableOp^dense_15/MatMul/ReadVariableOp ^dense_16/BiasAdd/ReadVariableOp^dense_16/MatMul/ReadVariableOp ^dense_17/BiasAdd/ReadVariableOp^dense_17/MatMul/ReadVariableOp ^dense_18/BiasAdd/ReadVariableOp^dense_18/MatMul/ReadVariableOp ^dense_19/BiasAdd/ReadVariableOp^dense_19/MatMul/ReadVariableOp ^dense_20/BiasAdd/ReadVariableOp^dense_20/MatMul/ReadVariableOp ^dense_21/BiasAdd/ReadVariableOp^dense_21/MatMul/ReadVariableOp ^dense_22/BiasAdd/ReadVariableOp^dense_22/MatMul/ReadVariableOp ^dense_23/BiasAdd/ReadVariableOp^dense_23/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity_1?

Identity_2Identityflatten_16/Reshape:output:0 ^conv3d_3/BiasAdd/ReadVariableOp^conv3d_3/Conv3D/ReadVariableOp ^conv3d_4/BiasAdd/ReadVariableOp^conv3d_4/Conv3D/ReadVariableOp ^dense_12/BiasAdd/ReadVariableOp^dense_12/MatMul/ReadVariableOp ^dense_13/BiasAdd/ReadVariableOp^dense_13/MatMul/ReadVariableOp ^dense_14/BiasAdd/ReadVariableOp^dense_14/MatMul/ReadVariableOp ^dense_15/BiasAdd/ReadVariableOp^dense_15/MatMul/ReadVariableOp ^dense_16/BiasAdd/ReadVariableOp^dense_16/MatMul/ReadVariableOp ^dense_17/BiasAdd/ReadVariableOp^dense_17/MatMul/ReadVariableOp ^dense_18/BiasAdd/ReadVariableOp^dense_18/MatMul/ReadVariableOp ^dense_19/BiasAdd/ReadVariableOp^dense_19/MatMul/ReadVariableOp ^dense_20/BiasAdd/ReadVariableOp^dense_20/MatMul/ReadVariableOp ^dense_21/BiasAdd/ReadVariableOp^dense_21/MatMul/ReadVariableOp ^dense_22/BiasAdd/ReadVariableOp^dense_22/MatMul/ReadVariableOp ^dense_23/BiasAdd/ReadVariableOp^dense_23/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity_2?

Identity_3Identityflatten_17/Reshape:output:0 ^conv3d_3/BiasAdd/ReadVariableOp^conv3d_3/Conv3D/ReadVariableOp ^conv3d_4/BiasAdd/ReadVariableOp^conv3d_4/Conv3D/ReadVariableOp ^dense_12/BiasAdd/ReadVariableOp^dense_12/MatMul/ReadVariableOp ^dense_13/BiasAdd/ReadVariableOp^dense_13/MatMul/ReadVariableOp ^dense_14/BiasAdd/ReadVariableOp^dense_14/MatMul/ReadVariableOp ^dense_15/BiasAdd/ReadVariableOp^dense_15/MatMul/ReadVariableOp ^dense_16/BiasAdd/ReadVariableOp^dense_16/MatMul/ReadVariableOp ^dense_17/BiasAdd/ReadVariableOp^dense_17/MatMul/ReadVariableOp ^dense_18/BiasAdd/ReadVariableOp^dense_18/MatMul/ReadVariableOp ^dense_19/BiasAdd/ReadVariableOp^dense_19/MatMul/ReadVariableOp ^dense_20/BiasAdd/ReadVariableOp^dense_20/MatMul/ReadVariableOp ^dense_21/BiasAdd/ReadVariableOp^dense_21/MatMul/ReadVariableOp ^dense_22/BiasAdd/ReadVariableOp^dense_22/MatMul/ReadVariableOp ^dense_23/BiasAdd/ReadVariableOp^dense_23/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity_3?

Identity_4Identityflatten_18/Reshape:output:0 ^conv3d_3/BiasAdd/ReadVariableOp^conv3d_3/Conv3D/ReadVariableOp ^conv3d_4/BiasAdd/ReadVariableOp^conv3d_4/Conv3D/ReadVariableOp ^dense_12/BiasAdd/ReadVariableOp^dense_12/MatMul/ReadVariableOp ^dense_13/BiasAdd/ReadVariableOp^dense_13/MatMul/ReadVariableOp ^dense_14/BiasAdd/ReadVariableOp^dense_14/MatMul/ReadVariableOp ^dense_15/BiasAdd/ReadVariableOp^dense_15/MatMul/ReadVariableOp ^dense_16/BiasAdd/ReadVariableOp^dense_16/MatMul/ReadVariableOp ^dense_17/BiasAdd/ReadVariableOp^dense_17/MatMul/ReadVariableOp ^dense_18/BiasAdd/ReadVariableOp^dense_18/MatMul/ReadVariableOp ^dense_19/BiasAdd/ReadVariableOp^dense_19/MatMul/ReadVariableOp ^dense_20/BiasAdd/ReadVariableOp^dense_20/MatMul/ReadVariableOp ^dense_21/BiasAdd/ReadVariableOp^dense_21/MatMul/ReadVariableOp ^dense_22/BiasAdd/ReadVariableOp^dense_22/MatMul/ReadVariableOp ^dense_23/BiasAdd/ReadVariableOp^dense_23/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity_4?

Identity_5Identityflatten_19/Reshape:output:0 ^conv3d_3/BiasAdd/ReadVariableOp^conv3d_3/Conv3D/ReadVariableOp ^conv3d_4/BiasAdd/ReadVariableOp^conv3d_4/Conv3D/ReadVariableOp ^dense_12/BiasAdd/ReadVariableOp^dense_12/MatMul/ReadVariableOp ^dense_13/BiasAdd/ReadVariableOp^dense_13/MatMul/ReadVariableOp ^dense_14/BiasAdd/ReadVariableOp^dense_14/MatMul/ReadVariableOp ^dense_15/BiasAdd/ReadVariableOp^dense_15/MatMul/ReadVariableOp ^dense_16/BiasAdd/ReadVariableOp^dense_16/MatMul/ReadVariableOp ^dense_17/BiasAdd/ReadVariableOp^dense_17/MatMul/ReadVariableOp ^dense_18/BiasAdd/ReadVariableOp^dense_18/MatMul/ReadVariableOp ^dense_19/BiasAdd/ReadVariableOp^dense_19/MatMul/ReadVariableOp ^dense_20/BiasAdd/ReadVariableOp^dense_20/MatMul/ReadVariableOp ^dense_21/BiasAdd/ReadVariableOp^dense_21/MatMul/ReadVariableOp ^dense_22/BiasAdd/ReadVariableOp^dense_22/MatMul/ReadVariableOp ^dense_23/BiasAdd/ReadVariableOp^dense_23/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity_5?

Identity_6Identityflatten_20/Reshape:output:0 ^conv3d_3/BiasAdd/ReadVariableOp^conv3d_3/Conv3D/ReadVariableOp ^conv3d_4/BiasAdd/ReadVariableOp^conv3d_4/Conv3D/ReadVariableOp ^dense_12/BiasAdd/ReadVariableOp^dense_12/MatMul/ReadVariableOp ^dense_13/BiasAdd/ReadVariableOp^dense_13/MatMul/ReadVariableOp ^dense_14/BiasAdd/ReadVariableOp^dense_14/MatMul/ReadVariableOp ^dense_15/BiasAdd/ReadVariableOp^dense_15/MatMul/ReadVariableOp ^dense_16/BiasAdd/ReadVariableOp^dense_16/MatMul/ReadVariableOp ^dense_17/BiasAdd/ReadVariableOp^dense_17/MatMul/ReadVariableOp ^dense_18/BiasAdd/ReadVariableOp^dense_18/MatMul/ReadVariableOp ^dense_19/BiasAdd/ReadVariableOp^dense_19/MatMul/ReadVariableOp ^dense_20/BiasAdd/ReadVariableOp^dense_20/MatMul/ReadVariableOp ^dense_21/BiasAdd/ReadVariableOp^dense_21/MatMul/ReadVariableOp ^dense_22/BiasAdd/ReadVariableOp^dense_22/MatMul/ReadVariableOp ^dense_23/BiasAdd/ReadVariableOp^dense_23/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity_6?

Identity_7Identityflatten_21/Reshape:output:0 ^conv3d_3/BiasAdd/ReadVariableOp^conv3d_3/Conv3D/ReadVariableOp ^conv3d_4/BiasAdd/ReadVariableOp^conv3d_4/Conv3D/ReadVariableOp ^dense_12/BiasAdd/ReadVariableOp^dense_12/MatMul/ReadVariableOp ^dense_13/BiasAdd/ReadVariableOp^dense_13/MatMul/ReadVariableOp ^dense_14/BiasAdd/ReadVariableOp^dense_14/MatMul/ReadVariableOp ^dense_15/BiasAdd/ReadVariableOp^dense_15/MatMul/ReadVariableOp ^dense_16/BiasAdd/ReadVariableOp^dense_16/MatMul/ReadVariableOp ^dense_17/BiasAdd/ReadVariableOp^dense_17/MatMul/ReadVariableOp ^dense_18/BiasAdd/ReadVariableOp^dense_18/MatMul/ReadVariableOp ^dense_19/BiasAdd/ReadVariableOp^dense_19/MatMul/ReadVariableOp ^dense_20/BiasAdd/ReadVariableOp^dense_20/MatMul/ReadVariableOp ^dense_21/BiasAdd/ReadVariableOp^dense_21/MatMul/ReadVariableOp ^dense_22/BiasAdd/ReadVariableOp^dense_22/MatMul/ReadVariableOp ^dense_23/BiasAdd/ReadVariableOp^dense_23/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity_7?

Identity_8Identityflatten_22/Reshape:output:0 ^conv3d_3/BiasAdd/ReadVariableOp^conv3d_3/Conv3D/ReadVariableOp ^conv3d_4/BiasAdd/ReadVariableOp^conv3d_4/Conv3D/ReadVariableOp ^dense_12/BiasAdd/ReadVariableOp^dense_12/MatMul/ReadVariableOp ^dense_13/BiasAdd/ReadVariableOp^dense_13/MatMul/ReadVariableOp ^dense_14/BiasAdd/ReadVariableOp^dense_14/MatMul/ReadVariableOp ^dense_15/BiasAdd/ReadVariableOp^dense_15/MatMul/ReadVariableOp ^dense_16/BiasAdd/ReadVariableOp^dense_16/MatMul/ReadVariableOp ^dense_17/BiasAdd/ReadVariableOp^dense_17/MatMul/ReadVariableOp ^dense_18/BiasAdd/ReadVariableOp^dense_18/MatMul/ReadVariableOp ^dense_19/BiasAdd/ReadVariableOp^dense_19/MatMul/ReadVariableOp ^dense_20/BiasAdd/ReadVariableOp^dense_20/MatMul/ReadVariableOp ^dense_21/BiasAdd/ReadVariableOp^dense_21/MatMul/ReadVariableOp ^dense_22/BiasAdd/ReadVariableOp^dense_22/MatMul/ReadVariableOp ^dense_23/BiasAdd/ReadVariableOp^dense_23/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity_8?

Identity_9Identityflatten_23/Reshape:output:0 ^conv3d_3/BiasAdd/ReadVariableOp^conv3d_3/Conv3D/ReadVariableOp ^conv3d_4/BiasAdd/ReadVariableOp^conv3d_4/Conv3D/ReadVariableOp ^dense_12/BiasAdd/ReadVariableOp^dense_12/MatMul/ReadVariableOp ^dense_13/BiasAdd/ReadVariableOp^dense_13/MatMul/ReadVariableOp ^dense_14/BiasAdd/ReadVariableOp^dense_14/MatMul/ReadVariableOp ^dense_15/BiasAdd/ReadVariableOp^dense_15/MatMul/ReadVariableOp ^dense_16/BiasAdd/ReadVariableOp^dense_16/MatMul/ReadVariableOp ^dense_17/BiasAdd/ReadVariableOp^dense_17/MatMul/ReadVariableOp ^dense_18/BiasAdd/ReadVariableOp^dense_18/MatMul/ReadVariableOp ^dense_19/BiasAdd/ReadVariableOp^dense_19/MatMul/ReadVariableOp ^dense_20/BiasAdd/ReadVariableOp^dense_20/MatMul/ReadVariableOp ^dense_21/BiasAdd/ReadVariableOp^dense_21/MatMul/ReadVariableOp ^dense_22/BiasAdd/ReadVariableOp^dense_22/MatMul/ReadVariableOp ^dense_23/BiasAdd/ReadVariableOp^dense_23/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity_9?
Identity_10Identityflatten_24/Reshape:output:0 ^conv3d_3/BiasAdd/ReadVariableOp^conv3d_3/Conv3D/ReadVariableOp ^conv3d_4/BiasAdd/ReadVariableOp^conv3d_4/Conv3D/ReadVariableOp ^dense_12/BiasAdd/ReadVariableOp^dense_12/MatMul/ReadVariableOp ^dense_13/BiasAdd/ReadVariableOp^dense_13/MatMul/ReadVariableOp ^dense_14/BiasAdd/ReadVariableOp^dense_14/MatMul/ReadVariableOp ^dense_15/BiasAdd/ReadVariableOp^dense_15/MatMul/ReadVariableOp ^dense_16/BiasAdd/ReadVariableOp^dense_16/MatMul/ReadVariableOp ^dense_17/BiasAdd/ReadVariableOp^dense_17/MatMul/ReadVariableOp ^dense_18/BiasAdd/ReadVariableOp^dense_18/MatMul/ReadVariableOp ^dense_19/BiasAdd/ReadVariableOp^dense_19/MatMul/ReadVariableOp ^dense_20/BiasAdd/ReadVariableOp^dense_20/MatMul/ReadVariableOp ^dense_21/BiasAdd/ReadVariableOp^dense_21/MatMul/ReadVariableOp ^dense_22/BiasAdd/ReadVariableOp^dense_22/MatMul/ReadVariableOp ^dense_23/BiasAdd/ReadVariableOp^dense_23/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2
Identity_10?
Identity_11Identityflatten_25/Reshape:output:0 ^conv3d_3/BiasAdd/ReadVariableOp^conv3d_3/Conv3D/ReadVariableOp ^conv3d_4/BiasAdd/ReadVariableOp^conv3d_4/Conv3D/ReadVariableOp ^dense_12/BiasAdd/ReadVariableOp^dense_12/MatMul/ReadVariableOp ^dense_13/BiasAdd/ReadVariableOp^dense_13/MatMul/ReadVariableOp ^dense_14/BiasAdd/ReadVariableOp^dense_14/MatMul/ReadVariableOp ^dense_15/BiasAdd/ReadVariableOp^dense_15/MatMul/ReadVariableOp ^dense_16/BiasAdd/ReadVariableOp^dense_16/MatMul/ReadVariableOp ^dense_17/BiasAdd/ReadVariableOp^dense_17/MatMul/ReadVariableOp ^dense_18/BiasAdd/ReadVariableOp^dense_18/MatMul/ReadVariableOp ^dense_19/BiasAdd/ReadVariableOp^dense_19/MatMul/ReadVariableOp ^dense_20/BiasAdd/ReadVariableOp^dense_20/MatMul/ReadVariableOp ^dense_21/BiasAdd/ReadVariableOp^dense_21/MatMul/ReadVariableOp ^dense_22/BiasAdd/ReadVariableOp^dense_22/MatMul/ReadVariableOp ^dense_23/BiasAdd/ReadVariableOp^dense_23/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2
Identity_11"
identityIdentity:output:0"!

identity_1Identity_1:output:0"#
identity_10Identity_10:output:0"#
identity_11Identity_11:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"!

identity_6Identity_6:output:0"!

identity_7Identity_7:output:0"!

identity_8Identity_8:output:0"!

identity_9Identity_9:output:0*?
_input_shapes?
?:?????????ppd::::::::::::::::::::::::::::2B
conv3d_3/BiasAdd/ReadVariableOpconv3d_3/BiasAdd/ReadVariableOp2@
conv3d_3/Conv3D/ReadVariableOpconv3d_3/Conv3D/ReadVariableOp2B
conv3d_4/BiasAdd/ReadVariableOpconv3d_4/BiasAdd/ReadVariableOp2@
conv3d_4/Conv3D/ReadVariableOpconv3d_4/Conv3D/ReadVariableOp2B
dense_12/BiasAdd/ReadVariableOpdense_12/BiasAdd/ReadVariableOp2@
dense_12/MatMul/ReadVariableOpdense_12/MatMul/ReadVariableOp2B
dense_13/BiasAdd/ReadVariableOpdense_13/BiasAdd/ReadVariableOp2@
dense_13/MatMul/ReadVariableOpdense_13/MatMul/ReadVariableOp2B
dense_14/BiasAdd/ReadVariableOpdense_14/BiasAdd/ReadVariableOp2@
dense_14/MatMul/ReadVariableOpdense_14/MatMul/ReadVariableOp2B
dense_15/BiasAdd/ReadVariableOpdense_15/BiasAdd/ReadVariableOp2@
dense_15/MatMul/ReadVariableOpdense_15/MatMul/ReadVariableOp2B
dense_16/BiasAdd/ReadVariableOpdense_16/BiasAdd/ReadVariableOp2@
dense_16/MatMul/ReadVariableOpdense_16/MatMul/ReadVariableOp2B
dense_17/BiasAdd/ReadVariableOpdense_17/BiasAdd/ReadVariableOp2@
dense_17/MatMul/ReadVariableOpdense_17/MatMul/ReadVariableOp2B
dense_18/BiasAdd/ReadVariableOpdense_18/BiasAdd/ReadVariableOp2@
dense_18/MatMul/ReadVariableOpdense_18/MatMul/ReadVariableOp2B
dense_19/BiasAdd/ReadVariableOpdense_19/BiasAdd/ReadVariableOp2@
dense_19/MatMul/ReadVariableOpdense_19/MatMul/ReadVariableOp2B
dense_20/BiasAdd/ReadVariableOpdense_20/BiasAdd/ReadVariableOp2@
dense_20/MatMul/ReadVariableOpdense_20/MatMul/ReadVariableOp2B
dense_21/BiasAdd/ReadVariableOpdense_21/BiasAdd/ReadVariableOp2@
dense_21/MatMul/ReadVariableOpdense_21/MatMul/ReadVariableOp2B
dense_22/BiasAdd/ReadVariableOpdense_22/BiasAdd/ReadVariableOp2@
dense_22/MatMul/ReadVariableOpdense_22/MatMul/ReadVariableOp2B
dense_23/BiasAdd/ReadVariableOpdense_23/BiasAdd/ReadVariableOp2@
dense_23/MatMul/ReadVariableOpdense_23/MatMul/ReadVariableOp:[ W
3
_output_shapes!
:?????????ppd
 
_user_specified_nameinputs
?	
?
B__inference_dense_12_layer_call_and_return_conditional_losses_8660

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
F
*__inference_flatten_25_layer_call_fn_10287

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_flatten_25_layer_call_and_return_conditional_losses_86822
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
`
D__inference_flatten_22_layer_call_and_return_conditional_losses_8724

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
Constg
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:?????????2	
Reshaped
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
C__inference_dense_20_layer_call_and_return_conditional_losses_10086

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
}
(__inference_dense_21_layer_call_fn_10115

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_21_layer_call_and_return_conditional_losses_84172
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
a
E__inference_flatten_14_layer_call_and_return_conditional_losses_10161

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
Constg
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:?????????2	
Reshaped
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
`
D__inference_flatten_23_layer_call_and_return_conditional_losses_8710

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
Constg
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:?????????2	
Reshaped
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
a
E__inference_flatten_18_layer_call_and_return_conditional_losses_10205

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
Constg
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:?????????2	
Reshaped
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?

?
B__inference_conv3d_3_layer_call_and_return_conditional_losses_8293

inputs"
conv3d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv3D/ReadVariableOp?
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:d*
dtype02
Conv3D/ReadVariableOp?
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????66*
paddingVALID*
strides	
2
Conv3D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????662	
BiasAddd
ReluReluBiasAdd:output:0*
T0*3
_output_shapes!
:?????????662
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*
T0*3
_output_shapes!
:?????????662

Identity"
identityIdentity:output:0*:
_input_shapes)
':?????????ppd::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:[ W
3
_output_shapes!
:?????????ppd
 
_user_specified_nameinputs
?	
?
B__inference_dense_18_layer_call_and_return_conditional_losses_8498

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
}
(__inference_dense_19_layer_call_fn_10075

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_19_layer_call_and_return_conditional_losses_84712
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
F
*__inference_flatten_14_layer_call_fn_10166

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_flatten_14_layer_call_and_return_conditional_losses_88362
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
??
?

A__inference_model_1_layer_call_and_return_conditional_losses_9059

inputs
conv3d_3_8962
conv3d_3_8964
conv3d_4_8968
conv3d_4_8970
dense_23_8975
dense_23_8977
dense_22_8980
dense_22_8982
dense_21_8985
dense_21_8987
dense_20_8990
dense_20_8992
dense_19_8995
dense_19_8997
dense_18_9000
dense_18_9002
dense_17_9005
dense_17_9007
dense_16_9010
dense_16_9012
dense_15_9015
dense_15_9017
dense_14_9020
dense_14_9022
dense_13_9025
dense_13_9027
dense_12_9030
dense_12_9032
identity

identity_1

identity_2

identity_3

identity_4

identity_5

identity_6

identity_7

identity_8

identity_9
identity_10
identity_11?? conv3d_3/StatefulPartitionedCall? conv3d_4/StatefulPartitionedCall? dense_12/StatefulPartitionedCall? dense_13/StatefulPartitionedCall? dense_14/StatefulPartitionedCall? dense_15/StatefulPartitionedCall? dense_16/StatefulPartitionedCall? dense_17/StatefulPartitionedCall? dense_18/StatefulPartitionedCall? dense_19/StatefulPartitionedCall? dense_20/StatefulPartitionedCall? dense_21/StatefulPartitionedCall? dense_22/StatefulPartitionedCall? dense_23/StatefulPartitionedCall?
 conv3d_3/StatefulPartitionedCallStatefulPartitionedCallinputsconv3d_3_8962conv3d_3_8964*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????66*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_conv3d_3_layer_call_and_return_conditional_losses_82932"
 conv3d_3/StatefulPartitionedCall?
max_pooling3d_2/PartitionedCallPartitionedCall)conv3d_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_max_pooling3d_2_layer_call_and_return_conditional_losses_82602!
max_pooling3d_2/PartitionedCall?
 conv3d_4/StatefulPartitionedCallStatefulPartitionedCall(max_pooling3d_2/PartitionedCall:output:0conv3d_4_8968conv3d_4_8970*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_conv3d_4_layer_call_and_return_conditional_losses_83212"
 conv3d_4/StatefulPartitionedCall?
max_pooling3d_3/PartitionedCallPartitionedCall)conv3d_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_max_pooling3d_3_layer_call_and_return_conditional_losses_82722!
max_pooling3d_3/PartitionedCall?
flatten_13/PartitionedCallPartitionedCall(max_pooling3d_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_flatten_13_layer_call_and_return_conditional_losses_83442
flatten_13/PartitionedCall?
 dense_23/StatefulPartitionedCallStatefulPartitionedCall#flatten_13/PartitionedCall:output:0dense_23_8975dense_23_8977*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_23_layer_call_and_return_conditional_losses_83632"
 dense_23/StatefulPartitionedCall?
 dense_22/StatefulPartitionedCallStatefulPartitionedCall#flatten_13/PartitionedCall:output:0dense_22_8980dense_22_8982*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_22_layer_call_and_return_conditional_losses_83902"
 dense_22/StatefulPartitionedCall?
 dense_21/StatefulPartitionedCallStatefulPartitionedCall#flatten_13/PartitionedCall:output:0dense_21_8985dense_21_8987*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_21_layer_call_and_return_conditional_losses_84172"
 dense_21/StatefulPartitionedCall?
 dense_20/StatefulPartitionedCallStatefulPartitionedCall#flatten_13/PartitionedCall:output:0dense_20_8990dense_20_8992*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_20_layer_call_and_return_conditional_losses_84442"
 dense_20/StatefulPartitionedCall?
 dense_19/StatefulPartitionedCallStatefulPartitionedCall#flatten_13/PartitionedCall:output:0dense_19_8995dense_19_8997*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_19_layer_call_and_return_conditional_losses_84712"
 dense_19/StatefulPartitionedCall?
 dense_18/StatefulPartitionedCallStatefulPartitionedCall#flatten_13/PartitionedCall:output:0dense_18_9000dense_18_9002*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_18_layer_call_and_return_conditional_losses_84982"
 dense_18/StatefulPartitionedCall?
 dense_17/StatefulPartitionedCallStatefulPartitionedCall#flatten_13/PartitionedCall:output:0dense_17_9005dense_17_9007*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_17_layer_call_and_return_conditional_losses_85252"
 dense_17/StatefulPartitionedCall?
 dense_16/StatefulPartitionedCallStatefulPartitionedCall#flatten_13/PartitionedCall:output:0dense_16_9010dense_16_9012*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_16_layer_call_and_return_conditional_losses_85522"
 dense_16/StatefulPartitionedCall?
 dense_15/StatefulPartitionedCallStatefulPartitionedCall#flatten_13/PartitionedCall:output:0dense_15_9015dense_15_9017*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_15_layer_call_and_return_conditional_losses_85792"
 dense_15/StatefulPartitionedCall?
 dense_14/StatefulPartitionedCallStatefulPartitionedCall#flatten_13/PartitionedCall:output:0dense_14_9020dense_14_9022*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_14_layer_call_and_return_conditional_losses_86062"
 dense_14/StatefulPartitionedCall?
 dense_13/StatefulPartitionedCallStatefulPartitionedCall#flatten_13/PartitionedCall:output:0dense_13_9025dense_13_9027*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_13_layer_call_and_return_conditional_losses_86332"
 dense_13/StatefulPartitionedCall?
 dense_12/StatefulPartitionedCallStatefulPartitionedCall#flatten_13/PartitionedCall:output:0dense_12_9030dense_12_9032*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_12_layer_call_and_return_conditional_losses_86602"
 dense_12/StatefulPartitionedCall?
flatten_25/PartitionedCallPartitionedCall)dense_23/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_flatten_25_layer_call_and_return_conditional_losses_86822
flatten_25/PartitionedCall?
flatten_24/PartitionedCallPartitionedCall)dense_22/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_flatten_24_layer_call_and_return_conditional_losses_86962
flatten_24/PartitionedCall?
flatten_23/PartitionedCallPartitionedCall)dense_21/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_flatten_23_layer_call_and_return_conditional_losses_87102
flatten_23/PartitionedCall?
flatten_22/PartitionedCallPartitionedCall)dense_20/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_flatten_22_layer_call_and_return_conditional_losses_87242
flatten_22/PartitionedCall?
flatten_21/PartitionedCallPartitionedCall)dense_19/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_flatten_21_layer_call_and_return_conditional_losses_87382
flatten_21/PartitionedCall?
flatten_20/PartitionedCallPartitionedCall)dense_18/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_flatten_20_layer_call_and_return_conditional_losses_87522
flatten_20/PartitionedCall?
flatten_19/PartitionedCallPartitionedCall)dense_17/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_flatten_19_layer_call_and_return_conditional_losses_87662
flatten_19/PartitionedCall?
flatten_18/PartitionedCallPartitionedCall)dense_16/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_flatten_18_layer_call_and_return_conditional_losses_87802
flatten_18/PartitionedCall?
flatten_17/PartitionedCallPartitionedCall)dense_15/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_flatten_17_layer_call_and_return_conditional_losses_87942
flatten_17/PartitionedCall?
flatten_16/PartitionedCallPartitionedCall)dense_14/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_flatten_16_layer_call_and_return_conditional_losses_88082
flatten_16/PartitionedCall?
flatten_15/PartitionedCallPartitionedCall)dense_13/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_flatten_15_layer_call_and_return_conditional_losses_88222
flatten_15/PartitionedCall?
flatten_14/PartitionedCallPartitionedCall)dense_12/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_flatten_14_layer_call_and_return_conditional_losses_88362
flatten_14/PartitionedCall?
IdentityIdentity#flatten_14/PartitionedCall:output:0!^conv3d_3/StatefulPartitionedCall!^conv3d_4/StatefulPartitionedCall!^dense_12/StatefulPartitionedCall!^dense_13/StatefulPartitionedCall!^dense_14/StatefulPartitionedCall!^dense_15/StatefulPartitionedCall!^dense_16/StatefulPartitionedCall!^dense_17/StatefulPartitionedCall!^dense_18/StatefulPartitionedCall!^dense_19/StatefulPartitionedCall!^dense_20/StatefulPartitionedCall!^dense_21/StatefulPartitionedCall!^dense_22/StatefulPartitionedCall!^dense_23/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity?

Identity_1Identity#flatten_15/PartitionedCall:output:0!^conv3d_3/StatefulPartitionedCall!^conv3d_4/StatefulPartitionedCall!^dense_12/StatefulPartitionedCall!^dense_13/StatefulPartitionedCall!^dense_14/StatefulPartitionedCall!^dense_15/StatefulPartitionedCall!^dense_16/StatefulPartitionedCall!^dense_17/StatefulPartitionedCall!^dense_18/StatefulPartitionedCall!^dense_19/StatefulPartitionedCall!^dense_20/StatefulPartitionedCall!^dense_21/StatefulPartitionedCall!^dense_22/StatefulPartitionedCall!^dense_23/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_1?

Identity_2Identity#flatten_16/PartitionedCall:output:0!^conv3d_3/StatefulPartitionedCall!^conv3d_4/StatefulPartitionedCall!^dense_12/StatefulPartitionedCall!^dense_13/StatefulPartitionedCall!^dense_14/StatefulPartitionedCall!^dense_15/StatefulPartitionedCall!^dense_16/StatefulPartitionedCall!^dense_17/StatefulPartitionedCall!^dense_18/StatefulPartitionedCall!^dense_19/StatefulPartitionedCall!^dense_20/StatefulPartitionedCall!^dense_21/StatefulPartitionedCall!^dense_22/StatefulPartitionedCall!^dense_23/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_2?

Identity_3Identity#flatten_17/PartitionedCall:output:0!^conv3d_3/StatefulPartitionedCall!^conv3d_4/StatefulPartitionedCall!^dense_12/StatefulPartitionedCall!^dense_13/StatefulPartitionedCall!^dense_14/StatefulPartitionedCall!^dense_15/StatefulPartitionedCall!^dense_16/StatefulPartitionedCall!^dense_17/StatefulPartitionedCall!^dense_18/StatefulPartitionedCall!^dense_19/StatefulPartitionedCall!^dense_20/StatefulPartitionedCall!^dense_21/StatefulPartitionedCall!^dense_22/StatefulPartitionedCall!^dense_23/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_3?

Identity_4Identity#flatten_18/PartitionedCall:output:0!^conv3d_3/StatefulPartitionedCall!^conv3d_4/StatefulPartitionedCall!^dense_12/StatefulPartitionedCall!^dense_13/StatefulPartitionedCall!^dense_14/StatefulPartitionedCall!^dense_15/StatefulPartitionedCall!^dense_16/StatefulPartitionedCall!^dense_17/StatefulPartitionedCall!^dense_18/StatefulPartitionedCall!^dense_19/StatefulPartitionedCall!^dense_20/StatefulPartitionedCall!^dense_21/StatefulPartitionedCall!^dense_22/StatefulPartitionedCall!^dense_23/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_4?

Identity_5Identity#flatten_19/PartitionedCall:output:0!^conv3d_3/StatefulPartitionedCall!^conv3d_4/StatefulPartitionedCall!^dense_12/StatefulPartitionedCall!^dense_13/StatefulPartitionedCall!^dense_14/StatefulPartitionedCall!^dense_15/StatefulPartitionedCall!^dense_16/StatefulPartitionedCall!^dense_17/StatefulPartitionedCall!^dense_18/StatefulPartitionedCall!^dense_19/StatefulPartitionedCall!^dense_20/StatefulPartitionedCall!^dense_21/StatefulPartitionedCall!^dense_22/StatefulPartitionedCall!^dense_23/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_5?

Identity_6Identity#flatten_20/PartitionedCall:output:0!^conv3d_3/StatefulPartitionedCall!^conv3d_4/StatefulPartitionedCall!^dense_12/StatefulPartitionedCall!^dense_13/StatefulPartitionedCall!^dense_14/StatefulPartitionedCall!^dense_15/StatefulPartitionedCall!^dense_16/StatefulPartitionedCall!^dense_17/StatefulPartitionedCall!^dense_18/StatefulPartitionedCall!^dense_19/StatefulPartitionedCall!^dense_20/StatefulPartitionedCall!^dense_21/StatefulPartitionedCall!^dense_22/StatefulPartitionedCall!^dense_23/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_6?

Identity_7Identity#flatten_21/PartitionedCall:output:0!^conv3d_3/StatefulPartitionedCall!^conv3d_4/StatefulPartitionedCall!^dense_12/StatefulPartitionedCall!^dense_13/StatefulPartitionedCall!^dense_14/StatefulPartitionedCall!^dense_15/StatefulPartitionedCall!^dense_16/StatefulPartitionedCall!^dense_17/StatefulPartitionedCall!^dense_18/StatefulPartitionedCall!^dense_19/StatefulPartitionedCall!^dense_20/StatefulPartitionedCall!^dense_21/StatefulPartitionedCall!^dense_22/StatefulPartitionedCall!^dense_23/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_7?

Identity_8Identity#flatten_22/PartitionedCall:output:0!^conv3d_3/StatefulPartitionedCall!^conv3d_4/StatefulPartitionedCall!^dense_12/StatefulPartitionedCall!^dense_13/StatefulPartitionedCall!^dense_14/StatefulPartitionedCall!^dense_15/StatefulPartitionedCall!^dense_16/StatefulPartitionedCall!^dense_17/StatefulPartitionedCall!^dense_18/StatefulPartitionedCall!^dense_19/StatefulPartitionedCall!^dense_20/StatefulPartitionedCall!^dense_21/StatefulPartitionedCall!^dense_22/StatefulPartitionedCall!^dense_23/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_8?

Identity_9Identity#flatten_23/PartitionedCall:output:0!^conv3d_3/StatefulPartitionedCall!^conv3d_4/StatefulPartitionedCall!^dense_12/StatefulPartitionedCall!^dense_13/StatefulPartitionedCall!^dense_14/StatefulPartitionedCall!^dense_15/StatefulPartitionedCall!^dense_16/StatefulPartitionedCall!^dense_17/StatefulPartitionedCall!^dense_18/StatefulPartitionedCall!^dense_19/StatefulPartitionedCall!^dense_20/StatefulPartitionedCall!^dense_21/StatefulPartitionedCall!^dense_22/StatefulPartitionedCall!^dense_23/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_9?
Identity_10Identity#flatten_24/PartitionedCall:output:0!^conv3d_3/StatefulPartitionedCall!^conv3d_4/StatefulPartitionedCall!^dense_12/StatefulPartitionedCall!^dense_13/StatefulPartitionedCall!^dense_14/StatefulPartitionedCall!^dense_15/StatefulPartitionedCall!^dense_16/StatefulPartitionedCall!^dense_17/StatefulPartitionedCall!^dense_18/StatefulPartitionedCall!^dense_19/StatefulPartitionedCall!^dense_20/StatefulPartitionedCall!^dense_21/StatefulPartitionedCall!^dense_22/StatefulPartitionedCall!^dense_23/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2
Identity_10?
Identity_11Identity#flatten_25/PartitionedCall:output:0!^conv3d_3/StatefulPartitionedCall!^conv3d_4/StatefulPartitionedCall!^dense_12/StatefulPartitionedCall!^dense_13/StatefulPartitionedCall!^dense_14/StatefulPartitionedCall!^dense_15/StatefulPartitionedCall!^dense_16/StatefulPartitionedCall!^dense_17/StatefulPartitionedCall!^dense_18/StatefulPartitionedCall!^dense_19/StatefulPartitionedCall!^dense_20/StatefulPartitionedCall!^dense_21/StatefulPartitionedCall!^dense_22/StatefulPartitionedCall!^dense_23/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2
Identity_11"
identityIdentity:output:0"!

identity_1Identity_1:output:0"#
identity_10Identity_10:output:0"#
identity_11Identity_11:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"!

identity_6Identity_6:output:0"!

identity_7Identity_7:output:0"!

identity_8Identity_8:output:0"!

identity_9Identity_9:output:0*?
_input_shapes?
?:?????????ppd::::::::::::::::::::::::::::2D
 conv3d_3/StatefulPartitionedCall conv3d_3/StatefulPartitionedCall2D
 conv3d_4/StatefulPartitionedCall conv3d_4/StatefulPartitionedCall2D
 dense_12/StatefulPartitionedCall dense_12/StatefulPartitionedCall2D
 dense_13/StatefulPartitionedCall dense_13/StatefulPartitionedCall2D
 dense_14/StatefulPartitionedCall dense_14/StatefulPartitionedCall2D
 dense_15/StatefulPartitionedCall dense_15/StatefulPartitionedCall2D
 dense_16/StatefulPartitionedCall dense_16/StatefulPartitionedCall2D
 dense_17/StatefulPartitionedCall dense_17/StatefulPartitionedCall2D
 dense_18/StatefulPartitionedCall dense_18/StatefulPartitionedCall2D
 dense_19/StatefulPartitionedCall dense_19/StatefulPartitionedCall2D
 dense_20/StatefulPartitionedCall dense_20/StatefulPartitionedCall2D
 dense_21/StatefulPartitionedCall dense_21/StatefulPartitionedCall2D
 dense_22/StatefulPartitionedCall dense_22/StatefulPartitionedCall2D
 dense_23/StatefulPartitionedCall dense_23/StatefulPartitionedCall:[ W
3
_output_shapes!
:?????????ppd
 
_user_specified_nameinputs
??
?

A__inference_model_1_layer_call_and_return_conditional_losses_8956
input_2
conv3d_3_8859
conv3d_3_8861
conv3d_4_8865
conv3d_4_8867
dense_23_8872
dense_23_8874
dense_22_8877
dense_22_8879
dense_21_8882
dense_21_8884
dense_20_8887
dense_20_8889
dense_19_8892
dense_19_8894
dense_18_8897
dense_18_8899
dense_17_8902
dense_17_8904
dense_16_8907
dense_16_8909
dense_15_8912
dense_15_8914
dense_14_8917
dense_14_8919
dense_13_8922
dense_13_8924
dense_12_8927
dense_12_8929
identity

identity_1

identity_2

identity_3

identity_4

identity_5

identity_6

identity_7

identity_8

identity_9
identity_10
identity_11?? conv3d_3/StatefulPartitionedCall? conv3d_4/StatefulPartitionedCall? dense_12/StatefulPartitionedCall? dense_13/StatefulPartitionedCall? dense_14/StatefulPartitionedCall? dense_15/StatefulPartitionedCall? dense_16/StatefulPartitionedCall? dense_17/StatefulPartitionedCall? dense_18/StatefulPartitionedCall? dense_19/StatefulPartitionedCall? dense_20/StatefulPartitionedCall? dense_21/StatefulPartitionedCall? dense_22/StatefulPartitionedCall? dense_23/StatefulPartitionedCall?
 conv3d_3/StatefulPartitionedCallStatefulPartitionedCallinput_2conv3d_3_8859conv3d_3_8861*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????66*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_conv3d_3_layer_call_and_return_conditional_losses_82932"
 conv3d_3/StatefulPartitionedCall?
max_pooling3d_2/PartitionedCallPartitionedCall)conv3d_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_max_pooling3d_2_layer_call_and_return_conditional_losses_82602!
max_pooling3d_2/PartitionedCall?
 conv3d_4/StatefulPartitionedCallStatefulPartitionedCall(max_pooling3d_2/PartitionedCall:output:0conv3d_4_8865conv3d_4_8867*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_conv3d_4_layer_call_and_return_conditional_losses_83212"
 conv3d_4/StatefulPartitionedCall?
max_pooling3d_3/PartitionedCallPartitionedCall)conv3d_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_max_pooling3d_3_layer_call_and_return_conditional_losses_82722!
max_pooling3d_3/PartitionedCall?
flatten_13/PartitionedCallPartitionedCall(max_pooling3d_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_flatten_13_layer_call_and_return_conditional_losses_83442
flatten_13/PartitionedCall?
 dense_23/StatefulPartitionedCallStatefulPartitionedCall#flatten_13/PartitionedCall:output:0dense_23_8872dense_23_8874*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_23_layer_call_and_return_conditional_losses_83632"
 dense_23/StatefulPartitionedCall?
 dense_22/StatefulPartitionedCallStatefulPartitionedCall#flatten_13/PartitionedCall:output:0dense_22_8877dense_22_8879*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_22_layer_call_and_return_conditional_losses_83902"
 dense_22/StatefulPartitionedCall?
 dense_21/StatefulPartitionedCallStatefulPartitionedCall#flatten_13/PartitionedCall:output:0dense_21_8882dense_21_8884*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_21_layer_call_and_return_conditional_losses_84172"
 dense_21/StatefulPartitionedCall?
 dense_20/StatefulPartitionedCallStatefulPartitionedCall#flatten_13/PartitionedCall:output:0dense_20_8887dense_20_8889*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_20_layer_call_and_return_conditional_losses_84442"
 dense_20/StatefulPartitionedCall?
 dense_19/StatefulPartitionedCallStatefulPartitionedCall#flatten_13/PartitionedCall:output:0dense_19_8892dense_19_8894*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_19_layer_call_and_return_conditional_losses_84712"
 dense_19/StatefulPartitionedCall?
 dense_18/StatefulPartitionedCallStatefulPartitionedCall#flatten_13/PartitionedCall:output:0dense_18_8897dense_18_8899*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_18_layer_call_and_return_conditional_losses_84982"
 dense_18/StatefulPartitionedCall?
 dense_17/StatefulPartitionedCallStatefulPartitionedCall#flatten_13/PartitionedCall:output:0dense_17_8902dense_17_8904*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_17_layer_call_and_return_conditional_losses_85252"
 dense_17/StatefulPartitionedCall?
 dense_16/StatefulPartitionedCallStatefulPartitionedCall#flatten_13/PartitionedCall:output:0dense_16_8907dense_16_8909*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_16_layer_call_and_return_conditional_losses_85522"
 dense_16/StatefulPartitionedCall?
 dense_15/StatefulPartitionedCallStatefulPartitionedCall#flatten_13/PartitionedCall:output:0dense_15_8912dense_15_8914*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_15_layer_call_and_return_conditional_losses_85792"
 dense_15/StatefulPartitionedCall?
 dense_14/StatefulPartitionedCallStatefulPartitionedCall#flatten_13/PartitionedCall:output:0dense_14_8917dense_14_8919*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_14_layer_call_and_return_conditional_losses_86062"
 dense_14/StatefulPartitionedCall?
 dense_13/StatefulPartitionedCallStatefulPartitionedCall#flatten_13/PartitionedCall:output:0dense_13_8922dense_13_8924*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_13_layer_call_and_return_conditional_losses_86332"
 dense_13/StatefulPartitionedCall?
 dense_12/StatefulPartitionedCallStatefulPartitionedCall#flatten_13/PartitionedCall:output:0dense_12_8927dense_12_8929*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_12_layer_call_and_return_conditional_losses_86602"
 dense_12/StatefulPartitionedCall?
flatten_25/PartitionedCallPartitionedCall)dense_23/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_flatten_25_layer_call_and_return_conditional_losses_86822
flatten_25/PartitionedCall?
flatten_24/PartitionedCallPartitionedCall)dense_22/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_flatten_24_layer_call_and_return_conditional_losses_86962
flatten_24/PartitionedCall?
flatten_23/PartitionedCallPartitionedCall)dense_21/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_flatten_23_layer_call_and_return_conditional_losses_87102
flatten_23/PartitionedCall?
flatten_22/PartitionedCallPartitionedCall)dense_20/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_flatten_22_layer_call_and_return_conditional_losses_87242
flatten_22/PartitionedCall?
flatten_21/PartitionedCallPartitionedCall)dense_19/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_flatten_21_layer_call_and_return_conditional_losses_87382
flatten_21/PartitionedCall?
flatten_20/PartitionedCallPartitionedCall)dense_18/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_flatten_20_layer_call_and_return_conditional_losses_87522
flatten_20/PartitionedCall?
flatten_19/PartitionedCallPartitionedCall)dense_17/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_flatten_19_layer_call_and_return_conditional_losses_87662
flatten_19/PartitionedCall?
flatten_18/PartitionedCallPartitionedCall)dense_16/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_flatten_18_layer_call_and_return_conditional_losses_87802
flatten_18/PartitionedCall?
flatten_17/PartitionedCallPartitionedCall)dense_15/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_flatten_17_layer_call_and_return_conditional_losses_87942
flatten_17/PartitionedCall?
flatten_16/PartitionedCallPartitionedCall)dense_14/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_flatten_16_layer_call_and_return_conditional_losses_88082
flatten_16/PartitionedCall?
flatten_15/PartitionedCallPartitionedCall)dense_13/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_flatten_15_layer_call_and_return_conditional_losses_88222
flatten_15/PartitionedCall?
flatten_14/PartitionedCallPartitionedCall)dense_12/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_flatten_14_layer_call_and_return_conditional_losses_88362
flatten_14/PartitionedCall?
IdentityIdentity#flatten_14/PartitionedCall:output:0!^conv3d_3/StatefulPartitionedCall!^conv3d_4/StatefulPartitionedCall!^dense_12/StatefulPartitionedCall!^dense_13/StatefulPartitionedCall!^dense_14/StatefulPartitionedCall!^dense_15/StatefulPartitionedCall!^dense_16/StatefulPartitionedCall!^dense_17/StatefulPartitionedCall!^dense_18/StatefulPartitionedCall!^dense_19/StatefulPartitionedCall!^dense_20/StatefulPartitionedCall!^dense_21/StatefulPartitionedCall!^dense_22/StatefulPartitionedCall!^dense_23/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity?

Identity_1Identity#flatten_15/PartitionedCall:output:0!^conv3d_3/StatefulPartitionedCall!^conv3d_4/StatefulPartitionedCall!^dense_12/StatefulPartitionedCall!^dense_13/StatefulPartitionedCall!^dense_14/StatefulPartitionedCall!^dense_15/StatefulPartitionedCall!^dense_16/StatefulPartitionedCall!^dense_17/StatefulPartitionedCall!^dense_18/StatefulPartitionedCall!^dense_19/StatefulPartitionedCall!^dense_20/StatefulPartitionedCall!^dense_21/StatefulPartitionedCall!^dense_22/StatefulPartitionedCall!^dense_23/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_1?

Identity_2Identity#flatten_16/PartitionedCall:output:0!^conv3d_3/StatefulPartitionedCall!^conv3d_4/StatefulPartitionedCall!^dense_12/StatefulPartitionedCall!^dense_13/StatefulPartitionedCall!^dense_14/StatefulPartitionedCall!^dense_15/StatefulPartitionedCall!^dense_16/StatefulPartitionedCall!^dense_17/StatefulPartitionedCall!^dense_18/StatefulPartitionedCall!^dense_19/StatefulPartitionedCall!^dense_20/StatefulPartitionedCall!^dense_21/StatefulPartitionedCall!^dense_22/StatefulPartitionedCall!^dense_23/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_2?

Identity_3Identity#flatten_17/PartitionedCall:output:0!^conv3d_3/StatefulPartitionedCall!^conv3d_4/StatefulPartitionedCall!^dense_12/StatefulPartitionedCall!^dense_13/StatefulPartitionedCall!^dense_14/StatefulPartitionedCall!^dense_15/StatefulPartitionedCall!^dense_16/StatefulPartitionedCall!^dense_17/StatefulPartitionedCall!^dense_18/StatefulPartitionedCall!^dense_19/StatefulPartitionedCall!^dense_20/StatefulPartitionedCall!^dense_21/StatefulPartitionedCall!^dense_22/StatefulPartitionedCall!^dense_23/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_3?

Identity_4Identity#flatten_18/PartitionedCall:output:0!^conv3d_3/StatefulPartitionedCall!^conv3d_4/StatefulPartitionedCall!^dense_12/StatefulPartitionedCall!^dense_13/StatefulPartitionedCall!^dense_14/StatefulPartitionedCall!^dense_15/StatefulPartitionedCall!^dense_16/StatefulPartitionedCall!^dense_17/StatefulPartitionedCall!^dense_18/StatefulPartitionedCall!^dense_19/StatefulPartitionedCall!^dense_20/StatefulPartitionedCall!^dense_21/StatefulPartitionedCall!^dense_22/StatefulPartitionedCall!^dense_23/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_4?

Identity_5Identity#flatten_19/PartitionedCall:output:0!^conv3d_3/StatefulPartitionedCall!^conv3d_4/StatefulPartitionedCall!^dense_12/StatefulPartitionedCall!^dense_13/StatefulPartitionedCall!^dense_14/StatefulPartitionedCall!^dense_15/StatefulPartitionedCall!^dense_16/StatefulPartitionedCall!^dense_17/StatefulPartitionedCall!^dense_18/StatefulPartitionedCall!^dense_19/StatefulPartitionedCall!^dense_20/StatefulPartitionedCall!^dense_21/StatefulPartitionedCall!^dense_22/StatefulPartitionedCall!^dense_23/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_5?

Identity_6Identity#flatten_20/PartitionedCall:output:0!^conv3d_3/StatefulPartitionedCall!^conv3d_4/StatefulPartitionedCall!^dense_12/StatefulPartitionedCall!^dense_13/StatefulPartitionedCall!^dense_14/StatefulPartitionedCall!^dense_15/StatefulPartitionedCall!^dense_16/StatefulPartitionedCall!^dense_17/StatefulPartitionedCall!^dense_18/StatefulPartitionedCall!^dense_19/StatefulPartitionedCall!^dense_20/StatefulPartitionedCall!^dense_21/StatefulPartitionedCall!^dense_22/StatefulPartitionedCall!^dense_23/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_6?

Identity_7Identity#flatten_21/PartitionedCall:output:0!^conv3d_3/StatefulPartitionedCall!^conv3d_4/StatefulPartitionedCall!^dense_12/StatefulPartitionedCall!^dense_13/StatefulPartitionedCall!^dense_14/StatefulPartitionedCall!^dense_15/StatefulPartitionedCall!^dense_16/StatefulPartitionedCall!^dense_17/StatefulPartitionedCall!^dense_18/StatefulPartitionedCall!^dense_19/StatefulPartitionedCall!^dense_20/StatefulPartitionedCall!^dense_21/StatefulPartitionedCall!^dense_22/StatefulPartitionedCall!^dense_23/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_7?

Identity_8Identity#flatten_22/PartitionedCall:output:0!^conv3d_3/StatefulPartitionedCall!^conv3d_4/StatefulPartitionedCall!^dense_12/StatefulPartitionedCall!^dense_13/StatefulPartitionedCall!^dense_14/StatefulPartitionedCall!^dense_15/StatefulPartitionedCall!^dense_16/StatefulPartitionedCall!^dense_17/StatefulPartitionedCall!^dense_18/StatefulPartitionedCall!^dense_19/StatefulPartitionedCall!^dense_20/StatefulPartitionedCall!^dense_21/StatefulPartitionedCall!^dense_22/StatefulPartitionedCall!^dense_23/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_8?

Identity_9Identity#flatten_23/PartitionedCall:output:0!^conv3d_3/StatefulPartitionedCall!^conv3d_4/StatefulPartitionedCall!^dense_12/StatefulPartitionedCall!^dense_13/StatefulPartitionedCall!^dense_14/StatefulPartitionedCall!^dense_15/StatefulPartitionedCall!^dense_16/StatefulPartitionedCall!^dense_17/StatefulPartitionedCall!^dense_18/StatefulPartitionedCall!^dense_19/StatefulPartitionedCall!^dense_20/StatefulPartitionedCall!^dense_21/StatefulPartitionedCall!^dense_22/StatefulPartitionedCall!^dense_23/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_9?
Identity_10Identity#flatten_24/PartitionedCall:output:0!^conv3d_3/StatefulPartitionedCall!^conv3d_4/StatefulPartitionedCall!^dense_12/StatefulPartitionedCall!^dense_13/StatefulPartitionedCall!^dense_14/StatefulPartitionedCall!^dense_15/StatefulPartitionedCall!^dense_16/StatefulPartitionedCall!^dense_17/StatefulPartitionedCall!^dense_18/StatefulPartitionedCall!^dense_19/StatefulPartitionedCall!^dense_20/StatefulPartitionedCall!^dense_21/StatefulPartitionedCall!^dense_22/StatefulPartitionedCall!^dense_23/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2
Identity_10?
Identity_11Identity#flatten_25/PartitionedCall:output:0!^conv3d_3/StatefulPartitionedCall!^conv3d_4/StatefulPartitionedCall!^dense_12/StatefulPartitionedCall!^dense_13/StatefulPartitionedCall!^dense_14/StatefulPartitionedCall!^dense_15/StatefulPartitionedCall!^dense_16/StatefulPartitionedCall!^dense_17/StatefulPartitionedCall!^dense_18/StatefulPartitionedCall!^dense_19/StatefulPartitionedCall!^dense_20/StatefulPartitionedCall!^dense_21/StatefulPartitionedCall!^dense_22/StatefulPartitionedCall!^dense_23/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2
Identity_11"
identityIdentity:output:0"!

identity_1Identity_1:output:0"#
identity_10Identity_10:output:0"#
identity_11Identity_11:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"!

identity_6Identity_6:output:0"!

identity_7Identity_7:output:0"!

identity_8Identity_8:output:0"!

identity_9Identity_9:output:0*?
_input_shapes?
?:?????????ppd::::::::::::::::::::::::::::2D
 conv3d_3/StatefulPartitionedCall conv3d_3/StatefulPartitionedCall2D
 conv3d_4/StatefulPartitionedCall conv3d_4/StatefulPartitionedCall2D
 dense_12/StatefulPartitionedCall dense_12/StatefulPartitionedCall2D
 dense_13/StatefulPartitionedCall dense_13/StatefulPartitionedCall2D
 dense_14/StatefulPartitionedCall dense_14/StatefulPartitionedCall2D
 dense_15/StatefulPartitionedCall dense_15/StatefulPartitionedCall2D
 dense_16/StatefulPartitionedCall dense_16/StatefulPartitionedCall2D
 dense_17/StatefulPartitionedCall dense_17/StatefulPartitionedCall2D
 dense_18/StatefulPartitionedCall dense_18/StatefulPartitionedCall2D
 dense_19/StatefulPartitionedCall dense_19/StatefulPartitionedCall2D
 dense_20/StatefulPartitionedCall dense_20/StatefulPartitionedCall2D
 dense_21/StatefulPartitionedCall dense_21/StatefulPartitionedCall2D
 dense_22/StatefulPartitionedCall dense_22/StatefulPartitionedCall2D
 dense_23/StatefulPartitionedCall dense_23/StatefulPartitionedCall:\ X
3
_output_shapes!
:?????????ppd
!
_user_specified_name	input_2
?
F
*__inference_flatten_15_layer_call_fn_10177

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_flatten_15_layer_call_and_return_conditional_losses_88222
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
? 
?
&__inference_model_1_layer_call_fn_9781

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26
identity

identity_1

identity_2

identity_3

identity_4

identity_5

identity_6

identity_7

identity_8

identity_9
identity_10
identity_11??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26*(
Tin!
2*
Tout
2*
_collective_manager_ids
 *?
_output_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????*>
_read_only_resource_inputs 
	
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_model_1_layer_call_and_return_conditional_losses_90592
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity?

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_1?

Identity_2Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_2?

Identity_3Identity StatefulPartitionedCall:output:3^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_3?

Identity_4Identity StatefulPartitionedCall:output:4^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_4?

Identity_5Identity StatefulPartitionedCall:output:5^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_5?

Identity_6Identity StatefulPartitionedCall:output:6^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_6?

Identity_7Identity StatefulPartitionedCall:output:7^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_7?

Identity_8Identity StatefulPartitionedCall:output:8^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_8?

Identity_9Identity StatefulPartitionedCall:output:9^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_9?
Identity_10Identity!StatefulPartitionedCall:output:10^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2
Identity_10?
Identity_11Identity!StatefulPartitionedCall:output:11^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2
Identity_11"
identityIdentity:output:0"!

identity_1Identity_1:output:0"#
identity_10Identity_10:output:0"#
identity_11Identity_11:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"!

identity_6Identity_6:output:0"!

identity_7Identity_7:output:0"!

identity_8Identity_8:output:0"!

identity_9Identity_9:output:0*?
_input_shapes?
?:?????????ppd::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:[ W
3
_output_shapes!
:?????????ppd
 
_user_specified_nameinputs
?
a
E__inference_flatten_25_layer_call_and_return_conditional_losses_10282

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
Constg
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:?????????2	
Reshaped
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
F
*__inference_flatten_21_layer_call_fn_10243

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_flatten_21_layer_call_and_return_conditional_losses_87382
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
? 
?
"__inference_signature_wrapper_9416
input_2
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26
identity

identity_1

identity_2

identity_3

identity_4

identity_5

identity_6

identity_7

identity_8

identity_9
identity_10
identity_11??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26*(
Tin!
2*
Tout
2*
_collective_manager_ids
 *?
_output_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????*>
_read_only_resource_inputs 
	
*-
config_proto

CPU

GPU 2J 8? *(
f#R!
__inference__wrapped_model_82542
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity?

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_1?

Identity_2Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_2?

Identity_3Identity StatefulPartitionedCall:output:3^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_3?

Identity_4Identity StatefulPartitionedCall:output:4^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_4?

Identity_5Identity StatefulPartitionedCall:output:5^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_5?

Identity_6Identity StatefulPartitionedCall:output:6^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_6?

Identity_7Identity StatefulPartitionedCall:output:7^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_7?

Identity_8Identity StatefulPartitionedCall:output:8^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_8?

Identity_9Identity StatefulPartitionedCall:output:9^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_9?
Identity_10Identity!StatefulPartitionedCall:output:10^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2
Identity_10?
Identity_11Identity!StatefulPartitionedCall:output:11^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2
Identity_11"
identityIdentity:output:0"!

identity_1Identity_1:output:0"#
identity_10Identity_10:output:0"#
identity_11Identity_11:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"!

identity_6Identity_6:output:0"!

identity_7Identity_7:output:0"!

identity_8Identity_8:output:0"!

identity_9Identity_9:output:0*?
_input_shapes?
?:?????????ppd::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:\ X
3
_output_shapes!
:?????????ppd
!
_user_specified_name	input_2
?
e
I__inference_max_pooling3d_2_layer_call_and_return_conditional_losses_8260

inputs
identity?
	MaxPool3D	MaxPool3Dinputs*
T0*W
_output_shapesE
C:A?????????????????????????????????????????????*
ksize	
*
paddingVALID*
strides	
2
	MaxPool3D?
IdentityIdentityMaxPool3D:output:0*
T0*W
_output_shapesE
C:A?????????????????????????????????????????????2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:A?????????????????????????????????????????????: {
W
_output_shapesE
C:A?????????????????????????????????????????????
 
_user_specified_nameinputs
?
`
D__inference_flatten_13_layer_call_and_return_conditional_losses_8344

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"????h  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:??????????2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*2
_input_shapes!
:?????????
:[ W
3
_output_shapes!
:?????????

 
_user_specified_nameinputs
?
`
D__inference_flatten_18_layer_call_and_return_conditional_losses_8780

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
Constg
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:?????????2	
Reshaped
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
??
?9
!__inference__traced_restore_11021
file_prefix$
 assignvariableop_conv3d_3_kernel$
 assignvariableop_1_conv3d_3_bias&
"assignvariableop_2_conv3d_4_kernel$
 assignvariableop_3_conv3d_4_bias&
"assignvariableop_4_dense_12_kernel$
 assignvariableop_5_dense_12_bias&
"assignvariableop_6_dense_13_kernel$
 assignvariableop_7_dense_13_bias&
"assignvariableop_8_dense_14_kernel$
 assignvariableop_9_dense_14_bias'
#assignvariableop_10_dense_15_kernel%
!assignvariableop_11_dense_15_bias'
#assignvariableop_12_dense_16_kernel%
!assignvariableop_13_dense_16_bias'
#assignvariableop_14_dense_17_kernel%
!assignvariableop_15_dense_17_bias'
#assignvariableop_16_dense_18_kernel%
!assignvariableop_17_dense_18_bias'
#assignvariableop_18_dense_19_kernel%
!assignvariableop_19_dense_19_bias'
#assignvariableop_20_dense_20_kernel%
!assignvariableop_21_dense_20_bias'
#assignvariableop_22_dense_21_kernel%
!assignvariableop_23_dense_21_bias'
#assignvariableop_24_dense_22_kernel%
!assignvariableop_25_dense_22_bias'
#assignvariableop_26_dense_23_kernel%
!assignvariableop_27_dense_23_bias!
assignvariableop_28_adam_iter#
assignvariableop_29_adam_beta_1#
assignvariableop_30_adam_beta_2"
assignvariableop_31_adam_decay*
&assignvariableop_32_adam_learning_rate
assignvariableop_33_total
assignvariableop_34_count
assignvariableop_35_total_1
assignvariableop_36_count_1
assignvariableop_37_total_2
assignvariableop_38_count_2
assignvariableop_39_total_3
assignvariableop_40_count_3
assignvariableop_41_total_4
assignvariableop_42_count_4
assignvariableop_43_total_5
assignvariableop_44_count_5
assignvariableop_45_total_6
assignvariableop_46_count_6
assignvariableop_47_total_7
assignvariableop_48_count_7
assignvariableop_49_total_8
assignvariableop_50_count_8
assignvariableop_51_total_9
assignvariableop_52_count_9 
assignvariableop_53_total_10 
assignvariableop_54_count_10 
assignvariableop_55_total_11 
assignvariableop_56_count_11 
assignvariableop_57_total_12 
assignvariableop_58_count_12.
*assignvariableop_59_adam_conv3d_3_kernel_m,
(assignvariableop_60_adam_conv3d_3_bias_m.
*assignvariableop_61_adam_conv3d_4_kernel_m,
(assignvariableop_62_adam_conv3d_4_bias_m.
*assignvariableop_63_adam_dense_12_kernel_m,
(assignvariableop_64_adam_dense_12_bias_m.
*assignvariableop_65_adam_dense_13_kernel_m,
(assignvariableop_66_adam_dense_13_bias_m.
*assignvariableop_67_adam_dense_14_kernel_m,
(assignvariableop_68_adam_dense_14_bias_m.
*assignvariableop_69_adam_dense_15_kernel_m,
(assignvariableop_70_adam_dense_15_bias_m.
*assignvariableop_71_adam_dense_16_kernel_m,
(assignvariableop_72_adam_dense_16_bias_m.
*assignvariableop_73_adam_dense_17_kernel_m,
(assignvariableop_74_adam_dense_17_bias_m.
*assignvariableop_75_adam_dense_18_kernel_m,
(assignvariableop_76_adam_dense_18_bias_m.
*assignvariableop_77_adam_dense_19_kernel_m,
(assignvariableop_78_adam_dense_19_bias_m.
*assignvariableop_79_adam_dense_20_kernel_m,
(assignvariableop_80_adam_dense_20_bias_m.
*assignvariableop_81_adam_dense_21_kernel_m,
(assignvariableop_82_adam_dense_21_bias_m.
*assignvariableop_83_adam_dense_22_kernel_m,
(assignvariableop_84_adam_dense_22_bias_m.
*assignvariableop_85_adam_dense_23_kernel_m,
(assignvariableop_86_adam_dense_23_bias_m.
*assignvariableop_87_adam_conv3d_3_kernel_v,
(assignvariableop_88_adam_conv3d_3_bias_v.
*assignvariableop_89_adam_conv3d_4_kernel_v,
(assignvariableop_90_adam_conv3d_4_bias_v.
*assignvariableop_91_adam_dense_12_kernel_v,
(assignvariableop_92_adam_dense_12_bias_v.
*assignvariableop_93_adam_dense_13_kernel_v,
(assignvariableop_94_adam_dense_13_bias_v.
*assignvariableop_95_adam_dense_14_kernel_v,
(assignvariableop_96_adam_dense_14_bias_v.
*assignvariableop_97_adam_dense_15_kernel_v,
(assignvariableop_98_adam_dense_15_bias_v.
*assignvariableop_99_adam_dense_16_kernel_v-
)assignvariableop_100_adam_dense_16_bias_v/
+assignvariableop_101_adam_dense_17_kernel_v-
)assignvariableop_102_adam_dense_17_bias_v/
+assignvariableop_103_adam_dense_18_kernel_v-
)assignvariableop_104_adam_dense_18_bias_v/
+assignvariableop_105_adam_dense_19_kernel_v-
)assignvariableop_106_adam_dense_19_bias_v/
+assignvariableop_107_adam_dense_20_kernel_v-
)assignvariableop_108_adam_dense_20_bias_v/
+assignvariableop_109_adam_dense_21_kernel_v-
)assignvariableop_110_adam_dense_21_bias_v/
+assignvariableop_111_adam_dense_22_kernel_v-
)assignvariableop_112_adam_dense_22_bias_v/
+assignvariableop_113_adam_dense_23_kernel_v-
)assignvariableop_114_adam_dense_23_bias_v
identity_116??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_100?AssignVariableOp_101?AssignVariableOp_102?AssignVariableOp_103?AssignVariableOp_104?AssignVariableOp_105?AssignVariableOp_106?AssignVariableOp_107?AssignVariableOp_108?AssignVariableOp_109?AssignVariableOp_11?AssignVariableOp_110?AssignVariableOp_111?AssignVariableOp_112?AssignVariableOp_113?AssignVariableOp_114?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_31?AssignVariableOp_32?AssignVariableOp_33?AssignVariableOp_34?AssignVariableOp_35?AssignVariableOp_36?AssignVariableOp_37?AssignVariableOp_38?AssignVariableOp_39?AssignVariableOp_4?AssignVariableOp_40?AssignVariableOp_41?AssignVariableOp_42?AssignVariableOp_43?AssignVariableOp_44?AssignVariableOp_45?AssignVariableOp_46?AssignVariableOp_47?AssignVariableOp_48?AssignVariableOp_49?AssignVariableOp_5?AssignVariableOp_50?AssignVariableOp_51?AssignVariableOp_52?AssignVariableOp_53?AssignVariableOp_54?AssignVariableOp_55?AssignVariableOp_56?AssignVariableOp_57?AssignVariableOp_58?AssignVariableOp_59?AssignVariableOp_6?AssignVariableOp_60?AssignVariableOp_61?AssignVariableOp_62?AssignVariableOp_63?AssignVariableOp_64?AssignVariableOp_65?AssignVariableOp_66?AssignVariableOp_67?AssignVariableOp_68?AssignVariableOp_69?AssignVariableOp_7?AssignVariableOp_70?AssignVariableOp_71?AssignVariableOp_72?AssignVariableOp_73?AssignVariableOp_74?AssignVariableOp_75?AssignVariableOp_76?AssignVariableOp_77?AssignVariableOp_78?AssignVariableOp_79?AssignVariableOp_8?AssignVariableOp_80?AssignVariableOp_81?AssignVariableOp_82?AssignVariableOp_83?AssignVariableOp_84?AssignVariableOp_85?AssignVariableOp_86?AssignVariableOp_87?AssignVariableOp_88?AssignVariableOp_89?AssignVariableOp_9?AssignVariableOp_90?AssignVariableOp_91?AssignVariableOp_92?AssignVariableOp_93?AssignVariableOp_94?AssignVariableOp_95?AssignVariableOp_96?AssignVariableOp_97?AssignVariableOp_98?AssignVariableOp_99?>
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:t*
dtype0*?=
value?=B?=tB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-13/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-13/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/3/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/3/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/4/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/4/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/5/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/5/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/6/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/6/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/7/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/7/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/8/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/8/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/9/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/9/count/.ATTRIBUTES/VARIABLE_VALUEB5keras_api/metrics/10/total/.ATTRIBUTES/VARIABLE_VALUEB5keras_api/metrics/10/count/.ATTRIBUTES/VARIABLE_VALUEB5keras_api/metrics/11/total/.ATTRIBUTES/VARIABLE_VALUEB5keras_api/metrics/11/count/.ATTRIBUTES/VARIABLE_VALUEB5keras_api/metrics/12/total/.ATTRIBUTES/VARIABLE_VALUEB5keras_api/metrics/12/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-12/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-13/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-13/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-12/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-13/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-13/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:t*
dtype0*?
value?B?tB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes?
?::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*?
dtypesx
v2t	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOp assignvariableop_conv3d_3_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOp assignvariableop_1_conv3d_3_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOp"assignvariableop_2_conv3d_4_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOp assignvariableop_3_conv3d_4_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOp"assignvariableop_4_dense_12_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOp assignvariableop_5_dense_12_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOp"assignvariableop_6_dense_13_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOp assignvariableop_7_dense_13_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOp"assignvariableop_8_dense_14_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOp assignvariableop_9_dense_14_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOp#assignvariableop_10_dense_15_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOp!assignvariableop_11_dense_15_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOp#assignvariableop_12_dense_16_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOp!assignvariableop_13_dense_16_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOp#assignvariableop_14_dense_17_kernelIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15?
AssignVariableOp_15AssignVariableOp!assignvariableop_15_dense_17_biasIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16?
AssignVariableOp_16AssignVariableOp#assignvariableop_16_dense_18_kernelIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17?
AssignVariableOp_17AssignVariableOp!assignvariableop_17_dense_18_biasIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18?
AssignVariableOp_18AssignVariableOp#assignvariableop_18_dense_19_kernelIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19?
AssignVariableOp_19AssignVariableOp!assignvariableop_19_dense_19_biasIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20?
AssignVariableOp_20AssignVariableOp#assignvariableop_20_dense_20_kernelIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21?
AssignVariableOp_21AssignVariableOp!assignvariableop_21_dense_20_biasIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22?
AssignVariableOp_22AssignVariableOp#assignvariableop_22_dense_21_kernelIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23?
AssignVariableOp_23AssignVariableOp!assignvariableop_23_dense_21_biasIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24?
AssignVariableOp_24AssignVariableOp#assignvariableop_24_dense_22_kernelIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25?
AssignVariableOp_25AssignVariableOp!assignvariableop_25_dense_22_biasIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26?
AssignVariableOp_26AssignVariableOp#assignvariableop_26_dense_23_kernelIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27?
AssignVariableOp_27AssignVariableOp!assignvariableop_27_dense_23_biasIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0	*
_output_shapes
:2
Identity_28?
AssignVariableOp_28AssignVariableOpassignvariableop_28_adam_iterIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29?
AssignVariableOp_29AssignVariableOpassignvariableop_29_adam_beta_1Identity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30?
AssignVariableOp_30AssignVariableOpassignvariableop_30_adam_beta_2Identity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31?
AssignVariableOp_31AssignVariableOpassignvariableop_31_adam_decayIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32?
AssignVariableOp_32AssignVariableOp&assignvariableop_32_adam_learning_rateIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33?
AssignVariableOp_33AssignVariableOpassignvariableop_33_totalIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34?
AssignVariableOp_34AssignVariableOpassignvariableop_34_countIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35?
AssignVariableOp_35AssignVariableOpassignvariableop_35_total_1Identity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36?
AssignVariableOp_36AssignVariableOpassignvariableop_36_count_1Identity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37?
AssignVariableOp_37AssignVariableOpassignvariableop_37_total_2Identity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38?
AssignVariableOp_38AssignVariableOpassignvariableop_38_count_2Identity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39?
AssignVariableOp_39AssignVariableOpassignvariableop_39_total_3Identity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40?
AssignVariableOp_40AssignVariableOpassignvariableop_40_count_3Identity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41?
AssignVariableOp_41AssignVariableOpassignvariableop_41_total_4Identity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42?
AssignVariableOp_42AssignVariableOpassignvariableop_42_count_4Identity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43?
AssignVariableOp_43AssignVariableOpassignvariableop_43_total_5Identity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44?
AssignVariableOp_44AssignVariableOpassignvariableop_44_count_5Identity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_44n
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:2
Identity_45?
AssignVariableOp_45AssignVariableOpassignvariableop_45_total_6Identity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_45n
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:2
Identity_46?
AssignVariableOp_46AssignVariableOpassignvariableop_46_count_6Identity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_46n
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:2
Identity_47?
AssignVariableOp_47AssignVariableOpassignvariableop_47_total_7Identity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_47n
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:2
Identity_48?
AssignVariableOp_48AssignVariableOpassignvariableop_48_count_7Identity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_48n
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:2
Identity_49?
AssignVariableOp_49AssignVariableOpassignvariableop_49_total_8Identity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_49n
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:2
Identity_50?
AssignVariableOp_50AssignVariableOpassignvariableop_50_count_8Identity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_50n
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:2
Identity_51?
AssignVariableOp_51AssignVariableOpassignvariableop_51_total_9Identity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_51n
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:2
Identity_52?
AssignVariableOp_52AssignVariableOpassignvariableop_52_count_9Identity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_52n
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:2
Identity_53?
AssignVariableOp_53AssignVariableOpassignvariableop_53_total_10Identity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_53n
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:2
Identity_54?
AssignVariableOp_54AssignVariableOpassignvariableop_54_count_10Identity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_54n
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:2
Identity_55?
AssignVariableOp_55AssignVariableOpassignvariableop_55_total_11Identity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_55n
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:2
Identity_56?
AssignVariableOp_56AssignVariableOpassignvariableop_56_count_11Identity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_56n
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:2
Identity_57?
AssignVariableOp_57AssignVariableOpassignvariableop_57_total_12Identity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_57n
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:2
Identity_58?
AssignVariableOp_58AssignVariableOpassignvariableop_58_count_12Identity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_58n
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:2
Identity_59?
AssignVariableOp_59AssignVariableOp*assignvariableop_59_adam_conv3d_3_kernel_mIdentity_59:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_59n
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:2
Identity_60?
AssignVariableOp_60AssignVariableOp(assignvariableop_60_adam_conv3d_3_bias_mIdentity_60:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_60n
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:2
Identity_61?
AssignVariableOp_61AssignVariableOp*assignvariableop_61_adam_conv3d_4_kernel_mIdentity_61:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_61n
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:2
Identity_62?
AssignVariableOp_62AssignVariableOp(assignvariableop_62_adam_conv3d_4_bias_mIdentity_62:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_62n
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:2
Identity_63?
AssignVariableOp_63AssignVariableOp*assignvariableop_63_adam_dense_12_kernel_mIdentity_63:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_63n
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:2
Identity_64?
AssignVariableOp_64AssignVariableOp(assignvariableop_64_adam_dense_12_bias_mIdentity_64:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_64n
Identity_65IdentityRestoreV2:tensors:65"/device:CPU:0*
T0*
_output_shapes
:2
Identity_65?
AssignVariableOp_65AssignVariableOp*assignvariableop_65_adam_dense_13_kernel_mIdentity_65:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_65n
Identity_66IdentityRestoreV2:tensors:66"/device:CPU:0*
T0*
_output_shapes
:2
Identity_66?
AssignVariableOp_66AssignVariableOp(assignvariableop_66_adam_dense_13_bias_mIdentity_66:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_66n
Identity_67IdentityRestoreV2:tensors:67"/device:CPU:0*
T0*
_output_shapes
:2
Identity_67?
AssignVariableOp_67AssignVariableOp*assignvariableop_67_adam_dense_14_kernel_mIdentity_67:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_67n
Identity_68IdentityRestoreV2:tensors:68"/device:CPU:0*
T0*
_output_shapes
:2
Identity_68?
AssignVariableOp_68AssignVariableOp(assignvariableop_68_adam_dense_14_bias_mIdentity_68:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_68n
Identity_69IdentityRestoreV2:tensors:69"/device:CPU:0*
T0*
_output_shapes
:2
Identity_69?
AssignVariableOp_69AssignVariableOp*assignvariableop_69_adam_dense_15_kernel_mIdentity_69:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_69n
Identity_70IdentityRestoreV2:tensors:70"/device:CPU:0*
T0*
_output_shapes
:2
Identity_70?
AssignVariableOp_70AssignVariableOp(assignvariableop_70_adam_dense_15_bias_mIdentity_70:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_70n
Identity_71IdentityRestoreV2:tensors:71"/device:CPU:0*
T0*
_output_shapes
:2
Identity_71?
AssignVariableOp_71AssignVariableOp*assignvariableop_71_adam_dense_16_kernel_mIdentity_71:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_71n
Identity_72IdentityRestoreV2:tensors:72"/device:CPU:0*
T0*
_output_shapes
:2
Identity_72?
AssignVariableOp_72AssignVariableOp(assignvariableop_72_adam_dense_16_bias_mIdentity_72:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_72n
Identity_73IdentityRestoreV2:tensors:73"/device:CPU:0*
T0*
_output_shapes
:2
Identity_73?
AssignVariableOp_73AssignVariableOp*assignvariableop_73_adam_dense_17_kernel_mIdentity_73:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_73n
Identity_74IdentityRestoreV2:tensors:74"/device:CPU:0*
T0*
_output_shapes
:2
Identity_74?
AssignVariableOp_74AssignVariableOp(assignvariableop_74_adam_dense_17_bias_mIdentity_74:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_74n
Identity_75IdentityRestoreV2:tensors:75"/device:CPU:0*
T0*
_output_shapes
:2
Identity_75?
AssignVariableOp_75AssignVariableOp*assignvariableop_75_adam_dense_18_kernel_mIdentity_75:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_75n
Identity_76IdentityRestoreV2:tensors:76"/device:CPU:0*
T0*
_output_shapes
:2
Identity_76?
AssignVariableOp_76AssignVariableOp(assignvariableop_76_adam_dense_18_bias_mIdentity_76:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_76n
Identity_77IdentityRestoreV2:tensors:77"/device:CPU:0*
T0*
_output_shapes
:2
Identity_77?
AssignVariableOp_77AssignVariableOp*assignvariableop_77_adam_dense_19_kernel_mIdentity_77:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_77n
Identity_78IdentityRestoreV2:tensors:78"/device:CPU:0*
T0*
_output_shapes
:2
Identity_78?
AssignVariableOp_78AssignVariableOp(assignvariableop_78_adam_dense_19_bias_mIdentity_78:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_78n
Identity_79IdentityRestoreV2:tensors:79"/device:CPU:0*
T0*
_output_shapes
:2
Identity_79?
AssignVariableOp_79AssignVariableOp*assignvariableop_79_adam_dense_20_kernel_mIdentity_79:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_79n
Identity_80IdentityRestoreV2:tensors:80"/device:CPU:0*
T0*
_output_shapes
:2
Identity_80?
AssignVariableOp_80AssignVariableOp(assignvariableop_80_adam_dense_20_bias_mIdentity_80:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_80n
Identity_81IdentityRestoreV2:tensors:81"/device:CPU:0*
T0*
_output_shapes
:2
Identity_81?
AssignVariableOp_81AssignVariableOp*assignvariableop_81_adam_dense_21_kernel_mIdentity_81:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_81n
Identity_82IdentityRestoreV2:tensors:82"/device:CPU:0*
T0*
_output_shapes
:2
Identity_82?
AssignVariableOp_82AssignVariableOp(assignvariableop_82_adam_dense_21_bias_mIdentity_82:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_82n
Identity_83IdentityRestoreV2:tensors:83"/device:CPU:0*
T0*
_output_shapes
:2
Identity_83?
AssignVariableOp_83AssignVariableOp*assignvariableop_83_adam_dense_22_kernel_mIdentity_83:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_83n
Identity_84IdentityRestoreV2:tensors:84"/device:CPU:0*
T0*
_output_shapes
:2
Identity_84?
AssignVariableOp_84AssignVariableOp(assignvariableop_84_adam_dense_22_bias_mIdentity_84:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_84n
Identity_85IdentityRestoreV2:tensors:85"/device:CPU:0*
T0*
_output_shapes
:2
Identity_85?
AssignVariableOp_85AssignVariableOp*assignvariableop_85_adam_dense_23_kernel_mIdentity_85:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_85n
Identity_86IdentityRestoreV2:tensors:86"/device:CPU:0*
T0*
_output_shapes
:2
Identity_86?
AssignVariableOp_86AssignVariableOp(assignvariableop_86_adam_dense_23_bias_mIdentity_86:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_86n
Identity_87IdentityRestoreV2:tensors:87"/device:CPU:0*
T0*
_output_shapes
:2
Identity_87?
AssignVariableOp_87AssignVariableOp*assignvariableop_87_adam_conv3d_3_kernel_vIdentity_87:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_87n
Identity_88IdentityRestoreV2:tensors:88"/device:CPU:0*
T0*
_output_shapes
:2
Identity_88?
AssignVariableOp_88AssignVariableOp(assignvariableop_88_adam_conv3d_3_bias_vIdentity_88:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_88n
Identity_89IdentityRestoreV2:tensors:89"/device:CPU:0*
T0*
_output_shapes
:2
Identity_89?
AssignVariableOp_89AssignVariableOp*assignvariableop_89_adam_conv3d_4_kernel_vIdentity_89:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_89n
Identity_90IdentityRestoreV2:tensors:90"/device:CPU:0*
T0*
_output_shapes
:2
Identity_90?
AssignVariableOp_90AssignVariableOp(assignvariableop_90_adam_conv3d_4_bias_vIdentity_90:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_90n
Identity_91IdentityRestoreV2:tensors:91"/device:CPU:0*
T0*
_output_shapes
:2
Identity_91?
AssignVariableOp_91AssignVariableOp*assignvariableop_91_adam_dense_12_kernel_vIdentity_91:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_91n
Identity_92IdentityRestoreV2:tensors:92"/device:CPU:0*
T0*
_output_shapes
:2
Identity_92?
AssignVariableOp_92AssignVariableOp(assignvariableop_92_adam_dense_12_bias_vIdentity_92:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_92n
Identity_93IdentityRestoreV2:tensors:93"/device:CPU:0*
T0*
_output_shapes
:2
Identity_93?
AssignVariableOp_93AssignVariableOp*assignvariableop_93_adam_dense_13_kernel_vIdentity_93:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_93n
Identity_94IdentityRestoreV2:tensors:94"/device:CPU:0*
T0*
_output_shapes
:2
Identity_94?
AssignVariableOp_94AssignVariableOp(assignvariableop_94_adam_dense_13_bias_vIdentity_94:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_94n
Identity_95IdentityRestoreV2:tensors:95"/device:CPU:0*
T0*
_output_shapes
:2
Identity_95?
AssignVariableOp_95AssignVariableOp*assignvariableop_95_adam_dense_14_kernel_vIdentity_95:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_95n
Identity_96IdentityRestoreV2:tensors:96"/device:CPU:0*
T0*
_output_shapes
:2
Identity_96?
AssignVariableOp_96AssignVariableOp(assignvariableop_96_adam_dense_14_bias_vIdentity_96:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_96n
Identity_97IdentityRestoreV2:tensors:97"/device:CPU:0*
T0*
_output_shapes
:2
Identity_97?
AssignVariableOp_97AssignVariableOp*assignvariableop_97_adam_dense_15_kernel_vIdentity_97:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_97n
Identity_98IdentityRestoreV2:tensors:98"/device:CPU:0*
T0*
_output_shapes
:2
Identity_98?
AssignVariableOp_98AssignVariableOp(assignvariableop_98_adam_dense_15_bias_vIdentity_98:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_98n
Identity_99IdentityRestoreV2:tensors:99"/device:CPU:0*
T0*
_output_shapes
:2
Identity_99?
AssignVariableOp_99AssignVariableOp*assignvariableop_99_adam_dense_16_kernel_vIdentity_99:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_99q
Identity_100IdentityRestoreV2:tensors:100"/device:CPU:0*
T0*
_output_shapes
:2
Identity_100?
AssignVariableOp_100AssignVariableOp)assignvariableop_100_adam_dense_16_bias_vIdentity_100:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_100q
Identity_101IdentityRestoreV2:tensors:101"/device:CPU:0*
T0*
_output_shapes
:2
Identity_101?
AssignVariableOp_101AssignVariableOp+assignvariableop_101_adam_dense_17_kernel_vIdentity_101:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_101q
Identity_102IdentityRestoreV2:tensors:102"/device:CPU:0*
T0*
_output_shapes
:2
Identity_102?
AssignVariableOp_102AssignVariableOp)assignvariableop_102_adam_dense_17_bias_vIdentity_102:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_102q
Identity_103IdentityRestoreV2:tensors:103"/device:CPU:0*
T0*
_output_shapes
:2
Identity_103?
AssignVariableOp_103AssignVariableOp+assignvariableop_103_adam_dense_18_kernel_vIdentity_103:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_103q
Identity_104IdentityRestoreV2:tensors:104"/device:CPU:0*
T0*
_output_shapes
:2
Identity_104?
AssignVariableOp_104AssignVariableOp)assignvariableop_104_adam_dense_18_bias_vIdentity_104:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_104q
Identity_105IdentityRestoreV2:tensors:105"/device:CPU:0*
T0*
_output_shapes
:2
Identity_105?
AssignVariableOp_105AssignVariableOp+assignvariableop_105_adam_dense_19_kernel_vIdentity_105:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_105q
Identity_106IdentityRestoreV2:tensors:106"/device:CPU:0*
T0*
_output_shapes
:2
Identity_106?
AssignVariableOp_106AssignVariableOp)assignvariableop_106_adam_dense_19_bias_vIdentity_106:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_106q
Identity_107IdentityRestoreV2:tensors:107"/device:CPU:0*
T0*
_output_shapes
:2
Identity_107?
AssignVariableOp_107AssignVariableOp+assignvariableop_107_adam_dense_20_kernel_vIdentity_107:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_107q
Identity_108IdentityRestoreV2:tensors:108"/device:CPU:0*
T0*
_output_shapes
:2
Identity_108?
AssignVariableOp_108AssignVariableOp)assignvariableop_108_adam_dense_20_bias_vIdentity_108:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_108q
Identity_109IdentityRestoreV2:tensors:109"/device:CPU:0*
T0*
_output_shapes
:2
Identity_109?
AssignVariableOp_109AssignVariableOp+assignvariableop_109_adam_dense_21_kernel_vIdentity_109:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_109q
Identity_110IdentityRestoreV2:tensors:110"/device:CPU:0*
T0*
_output_shapes
:2
Identity_110?
AssignVariableOp_110AssignVariableOp)assignvariableop_110_adam_dense_21_bias_vIdentity_110:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_110q
Identity_111IdentityRestoreV2:tensors:111"/device:CPU:0*
T0*
_output_shapes
:2
Identity_111?
AssignVariableOp_111AssignVariableOp+assignvariableop_111_adam_dense_22_kernel_vIdentity_111:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_111q
Identity_112IdentityRestoreV2:tensors:112"/device:CPU:0*
T0*
_output_shapes
:2
Identity_112?
AssignVariableOp_112AssignVariableOp)assignvariableop_112_adam_dense_22_bias_vIdentity_112:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_112q
Identity_113IdentityRestoreV2:tensors:113"/device:CPU:0*
T0*
_output_shapes
:2
Identity_113?
AssignVariableOp_113AssignVariableOp+assignvariableop_113_adam_dense_23_kernel_vIdentity_113:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_113q
Identity_114IdentityRestoreV2:tensors:114"/device:CPU:0*
T0*
_output_shapes
:2
Identity_114?
AssignVariableOp_114AssignVariableOp)assignvariableop_114_adam_dense_23_bias_vIdentity_114:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1149
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?
Identity_115Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_100^AssignVariableOp_101^AssignVariableOp_102^AssignVariableOp_103^AssignVariableOp_104^AssignVariableOp_105^AssignVariableOp_106^AssignVariableOp_107^AssignVariableOp_108^AssignVariableOp_109^AssignVariableOp_11^AssignVariableOp_110^AssignVariableOp_111^AssignVariableOp_112^AssignVariableOp_113^AssignVariableOp_114^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_87^AssignVariableOp_88^AssignVariableOp_89^AssignVariableOp_9^AssignVariableOp_90^AssignVariableOp_91^AssignVariableOp_92^AssignVariableOp_93^AssignVariableOp_94^AssignVariableOp_95^AssignVariableOp_96^AssignVariableOp_97^AssignVariableOp_98^AssignVariableOp_99^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_115?
Identity_116IdentityIdentity_115:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_100^AssignVariableOp_101^AssignVariableOp_102^AssignVariableOp_103^AssignVariableOp_104^AssignVariableOp_105^AssignVariableOp_106^AssignVariableOp_107^AssignVariableOp_108^AssignVariableOp_109^AssignVariableOp_11^AssignVariableOp_110^AssignVariableOp_111^AssignVariableOp_112^AssignVariableOp_113^AssignVariableOp_114^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_87^AssignVariableOp_88^AssignVariableOp_89^AssignVariableOp_9^AssignVariableOp_90^AssignVariableOp_91^AssignVariableOp_92^AssignVariableOp_93^AssignVariableOp_94^AssignVariableOp_95^AssignVariableOp_96^AssignVariableOp_97^AssignVariableOp_98^AssignVariableOp_99*
T0*
_output_shapes
: 2
Identity_116"%
identity_116Identity_116:output:0*?
_input_shapes?
?: :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102,
AssignVariableOp_100AssignVariableOp_1002,
AssignVariableOp_101AssignVariableOp_1012,
AssignVariableOp_102AssignVariableOp_1022,
AssignVariableOp_103AssignVariableOp_1032,
AssignVariableOp_104AssignVariableOp_1042,
AssignVariableOp_105AssignVariableOp_1052,
AssignVariableOp_106AssignVariableOp_1062,
AssignVariableOp_107AssignVariableOp_1072,
AssignVariableOp_108AssignVariableOp_1082,
AssignVariableOp_109AssignVariableOp_1092*
AssignVariableOp_11AssignVariableOp_112,
AssignVariableOp_110AssignVariableOp_1102,
AssignVariableOp_111AssignVariableOp_1112,
AssignVariableOp_112AssignVariableOp_1122,
AssignVariableOp_113AssignVariableOp_1132,
AssignVariableOp_114AssignVariableOp_1142*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_54AssignVariableOp_542*
AssignVariableOp_55AssignVariableOp_552*
AssignVariableOp_56AssignVariableOp_562*
AssignVariableOp_57AssignVariableOp_572*
AssignVariableOp_58AssignVariableOp_582*
AssignVariableOp_59AssignVariableOp_592(
AssignVariableOp_6AssignVariableOp_62*
AssignVariableOp_60AssignVariableOp_602*
AssignVariableOp_61AssignVariableOp_612*
AssignVariableOp_62AssignVariableOp_622*
AssignVariableOp_63AssignVariableOp_632*
AssignVariableOp_64AssignVariableOp_642*
AssignVariableOp_65AssignVariableOp_652*
AssignVariableOp_66AssignVariableOp_662*
AssignVariableOp_67AssignVariableOp_672*
AssignVariableOp_68AssignVariableOp_682*
AssignVariableOp_69AssignVariableOp_692(
AssignVariableOp_7AssignVariableOp_72*
AssignVariableOp_70AssignVariableOp_702*
AssignVariableOp_71AssignVariableOp_712*
AssignVariableOp_72AssignVariableOp_722*
AssignVariableOp_73AssignVariableOp_732*
AssignVariableOp_74AssignVariableOp_742*
AssignVariableOp_75AssignVariableOp_752*
AssignVariableOp_76AssignVariableOp_762*
AssignVariableOp_77AssignVariableOp_772*
AssignVariableOp_78AssignVariableOp_782*
AssignVariableOp_79AssignVariableOp_792(
AssignVariableOp_8AssignVariableOp_82*
AssignVariableOp_80AssignVariableOp_802*
AssignVariableOp_81AssignVariableOp_812*
AssignVariableOp_82AssignVariableOp_822*
AssignVariableOp_83AssignVariableOp_832*
AssignVariableOp_84AssignVariableOp_842*
AssignVariableOp_85AssignVariableOp_852*
AssignVariableOp_86AssignVariableOp_862*
AssignVariableOp_87AssignVariableOp_872*
AssignVariableOp_88AssignVariableOp_882*
AssignVariableOp_89AssignVariableOp_892(
AssignVariableOp_9AssignVariableOp_92*
AssignVariableOp_90AssignVariableOp_902*
AssignVariableOp_91AssignVariableOp_912*
AssignVariableOp_92AssignVariableOp_922*
AssignVariableOp_93AssignVariableOp_932*
AssignVariableOp_94AssignVariableOp_942*
AssignVariableOp_95AssignVariableOp_952*
AssignVariableOp_96AssignVariableOp_962*
AssignVariableOp_97AssignVariableOp_972*
AssignVariableOp_98AssignVariableOp_982*
AssignVariableOp_99AssignVariableOp_99:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?	
?
C__inference_dense_22_layer_call_and_return_conditional_losses_10126

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
`
D__inference_flatten_15_layer_call_and_return_conditional_losses_8822

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
Constg
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:?????????2	
Reshaped
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?

?
B__inference_conv3d_4_layer_call_and_return_conditional_losses_8321

inputs"
conv3d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv3D/ReadVariableOp?
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:
*
dtype02
Conv3D/ReadVariableOp?
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????
*
paddingVALID*
strides	
2
Conv3D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????
2	
BiasAddd
ReluReluBiasAdd:output:0*
T0*3
_output_shapes!
:?????????
2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*
T0*3
_output_shapes!
:?????????
2

Identity"
identityIdentity:output:0*:
_input_shapes)
':?????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:[ W
3
_output_shapes!
:?????????
 
_user_specified_nameinputs
?
|
'__inference_dense_13_layer_call_fn_9955

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_13_layer_call_and_return_conditional_losses_86332
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
`
D__inference_flatten_25_layer_call_and_return_conditional_losses_8682

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
Constg
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:?????????2	
Reshaped
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
J
.__inference_max_pooling3d_2_layer_call_fn_8266

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *W
_output_shapesE
C:A?????????????????????????????????????????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_max_pooling3d_2_layer_call_and_return_conditional_losses_82602
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*W
_output_shapesE
C:A?????????????????????????????????????????????2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:A?????????????????????????????????????????????: {
W
_output_shapesE
C:A?????????????????????????????????????????????
 
_user_specified_nameinputs
?	
?
C__inference_dense_18_layer_call_and_return_conditional_losses_10046

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
a
E__inference_flatten_20_layer_call_and_return_conditional_losses_10227

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
Constg
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:?????????2	
Reshaped
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
|
'__inference_conv3d_3_layer_call_fn_9884

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????66*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_conv3d_3_layer_call_and_return_conditional_losses_82932
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*3
_output_shapes!
:?????????662

Identity"
identityIdentity:output:0*:
_input_shapes)
':?????????ppd::22
StatefulPartitionedCallStatefulPartitionedCall:[ W
3
_output_shapes!
:?????????ppd
 
_user_specified_nameinputs
? 
?
&__inference_model_1_layer_call_fn_9864

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26
identity

identity_1

identity_2

identity_3

identity_4

identity_5

identity_6

identity_7

identity_8

identity_9
identity_10
identity_11??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26*(
Tin!
2*
Tout
2*
_collective_manager_ids
 *?
_output_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????*>
_read_only_resource_inputs 
	
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_model_1_layer_call_and_return_conditional_losses_92422
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity?

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_1?

Identity_2Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_2?

Identity_3Identity StatefulPartitionedCall:output:3^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_3?

Identity_4Identity StatefulPartitionedCall:output:4^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_4?

Identity_5Identity StatefulPartitionedCall:output:5^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_5?

Identity_6Identity StatefulPartitionedCall:output:6^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_6?

Identity_7Identity StatefulPartitionedCall:output:7^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_7?

Identity_8Identity StatefulPartitionedCall:output:8^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_8?

Identity_9Identity StatefulPartitionedCall:output:9^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_9?
Identity_10Identity!StatefulPartitionedCall:output:10^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2
Identity_10?
Identity_11Identity!StatefulPartitionedCall:output:11^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2
Identity_11"
identityIdentity:output:0"!

identity_1Identity_1:output:0"#
identity_10Identity_10:output:0"#
identity_11Identity_11:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"!

identity_6Identity_6:output:0"!

identity_7Identity_7:output:0"!

identity_8Identity_8:output:0"!

identity_9Identity_9:output:0*?
_input_shapes?
?:?????????ppd::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:[ W
3
_output_shapes!
:?????????ppd
 
_user_specified_nameinputs
??
?
A__inference_model_1_layer_call_and_return_conditional_losses_9698

inputs+
'conv3d_3_conv3d_readvariableop_resource,
(conv3d_3_biasadd_readvariableop_resource+
'conv3d_4_conv3d_readvariableop_resource,
(conv3d_4_biasadd_readvariableop_resource+
'dense_23_matmul_readvariableop_resource,
(dense_23_biasadd_readvariableop_resource+
'dense_22_matmul_readvariableop_resource,
(dense_22_biasadd_readvariableop_resource+
'dense_21_matmul_readvariableop_resource,
(dense_21_biasadd_readvariableop_resource+
'dense_20_matmul_readvariableop_resource,
(dense_20_biasadd_readvariableop_resource+
'dense_19_matmul_readvariableop_resource,
(dense_19_biasadd_readvariableop_resource+
'dense_18_matmul_readvariableop_resource,
(dense_18_biasadd_readvariableop_resource+
'dense_17_matmul_readvariableop_resource,
(dense_17_biasadd_readvariableop_resource+
'dense_16_matmul_readvariableop_resource,
(dense_16_biasadd_readvariableop_resource+
'dense_15_matmul_readvariableop_resource,
(dense_15_biasadd_readvariableop_resource+
'dense_14_matmul_readvariableop_resource,
(dense_14_biasadd_readvariableop_resource+
'dense_13_matmul_readvariableop_resource,
(dense_13_biasadd_readvariableop_resource+
'dense_12_matmul_readvariableop_resource,
(dense_12_biasadd_readvariableop_resource
identity

identity_1

identity_2

identity_3

identity_4

identity_5

identity_6

identity_7

identity_8

identity_9
identity_10
identity_11??conv3d_3/BiasAdd/ReadVariableOp?conv3d_3/Conv3D/ReadVariableOp?conv3d_4/BiasAdd/ReadVariableOp?conv3d_4/Conv3D/ReadVariableOp?dense_12/BiasAdd/ReadVariableOp?dense_12/MatMul/ReadVariableOp?dense_13/BiasAdd/ReadVariableOp?dense_13/MatMul/ReadVariableOp?dense_14/BiasAdd/ReadVariableOp?dense_14/MatMul/ReadVariableOp?dense_15/BiasAdd/ReadVariableOp?dense_15/MatMul/ReadVariableOp?dense_16/BiasAdd/ReadVariableOp?dense_16/MatMul/ReadVariableOp?dense_17/BiasAdd/ReadVariableOp?dense_17/MatMul/ReadVariableOp?dense_18/BiasAdd/ReadVariableOp?dense_18/MatMul/ReadVariableOp?dense_19/BiasAdd/ReadVariableOp?dense_19/MatMul/ReadVariableOp?dense_20/BiasAdd/ReadVariableOp?dense_20/MatMul/ReadVariableOp?dense_21/BiasAdd/ReadVariableOp?dense_21/MatMul/ReadVariableOp?dense_22/BiasAdd/ReadVariableOp?dense_22/MatMul/ReadVariableOp?dense_23/BiasAdd/ReadVariableOp?dense_23/MatMul/ReadVariableOp?
conv3d_3/Conv3D/ReadVariableOpReadVariableOp'conv3d_3_conv3d_readvariableop_resource**
_output_shapes
:d*
dtype02 
conv3d_3/Conv3D/ReadVariableOp?
conv3d_3/Conv3DConv3Dinputs&conv3d_3/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????66*
paddingVALID*
strides	
2
conv3d_3/Conv3D?
conv3d_3/BiasAdd/ReadVariableOpReadVariableOp(conv3d_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
conv3d_3/BiasAdd/ReadVariableOp?
conv3d_3/BiasAddBiasAddconv3d_3/Conv3D:output:0'conv3d_3/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????662
conv3d_3/BiasAdd
conv3d_3/ReluReluconv3d_3/BiasAdd:output:0*
T0*3
_output_shapes!
:?????????662
conv3d_3/Relu?
max_pooling3d_2/MaxPool3D	MaxPool3Dconv3d_3/Relu:activations:0*
T0*3
_output_shapes!
:?????????*
ksize	
*
paddingVALID*
strides	
2
max_pooling3d_2/MaxPool3D?
conv3d_4/Conv3D/ReadVariableOpReadVariableOp'conv3d_4_conv3d_readvariableop_resource**
_output_shapes
:
*
dtype02 
conv3d_4/Conv3D/ReadVariableOp?
conv3d_4/Conv3DConv3D"max_pooling3d_2/MaxPool3D:output:0&conv3d_4/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????
*
paddingVALID*
strides	
2
conv3d_4/Conv3D?
conv3d_4/BiasAdd/ReadVariableOpReadVariableOp(conv3d_4_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02!
conv3d_4/BiasAdd/ReadVariableOp?
conv3d_4/BiasAddBiasAddconv3d_4/Conv3D:output:0'conv3d_4/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????
2
conv3d_4/BiasAdd
conv3d_4/ReluReluconv3d_4/BiasAdd:output:0*
T0*3
_output_shapes!
:?????????
2
conv3d_4/Relu?
max_pooling3d_3/MaxPool3D	MaxPool3Dconv3d_4/Relu:activations:0*
T0*3
_output_shapes!
:?????????
*
ksize	
*
paddingVALID*
strides	
2
max_pooling3d_3/MaxPool3Du
flatten_13/ConstConst*
_output_shapes
:*
dtype0*
valueB"????h  2
flatten_13/Const?
flatten_13/ReshapeReshape"max_pooling3d_3/MaxPool3D:output:0flatten_13/Const:output:0*
T0*(
_output_shapes
:??????????2
flatten_13/Reshape?
dense_23/MatMul/ReadVariableOpReadVariableOp'dense_23_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02 
dense_23/MatMul/ReadVariableOp?
dense_23/MatMulMatMulflatten_13/Reshape:output:0&dense_23/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_23/MatMul?
dense_23/BiasAdd/ReadVariableOpReadVariableOp(dense_23_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_23/BiasAdd/ReadVariableOp?
dense_23/BiasAddBiasAdddense_23/MatMul:product:0'dense_23/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_23/BiasAdd|
dense_23/SigmoidSigmoiddense_23/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_23/Sigmoid?
dense_22/MatMul/ReadVariableOpReadVariableOp'dense_22_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02 
dense_22/MatMul/ReadVariableOp?
dense_22/MatMulMatMulflatten_13/Reshape:output:0&dense_22/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_22/MatMul?
dense_22/BiasAdd/ReadVariableOpReadVariableOp(dense_22_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_22/BiasAdd/ReadVariableOp?
dense_22/BiasAddBiasAdddense_22/MatMul:product:0'dense_22/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_22/BiasAdds
dense_22/ReluReludense_22/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_22/Relu?
dense_21/MatMul/ReadVariableOpReadVariableOp'dense_21_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02 
dense_21/MatMul/ReadVariableOp?
dense_21/MatMulMatMulflatten_13/Reshape:output:0&dense_21/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_21/MatMul?
dense_21/BiasAdd/ReadVariableOpReadVariableOp(dense_21_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_21/BiasAdd/ReadVariableOp?
dense_21/BiasAddBiasAdddense_21/MatMul:product:0'dense_21/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_21/BiasAdds
dense_21/ReluReludense_21/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_21/Relu?
dense_20/MatMul/ReadVariableOpReadVariableOp'dense_20_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02 
dense_20/MatMul/ReadVariableOp?
dense_20/MatMulMatMulflatten_13/Reshape:output:0&dense_20/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_20/MatMul?
dense_20/BiasAdd/ReadVariableOpReadVariableOp(dense_20_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_20/BiasAdd/ReadVariableOp?
dense_20/BiasAddBiasAdddense_20/MatMul:product:0'dense_20/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_20/BiasAdds
dense_20/ReluReludense_20/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_20/Relu?
dense_19/MatMul/ReadVariableOpReadVariableOp'dense_19_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02 
dense_19/MatMul/ReadVariableOp?
dense_19/MatMulMatMulflatten_13/Reshape:output:0&dense_19/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_19/MatMul?
dense_19/BiasAdd/ReadVariableOpReadVariableOp(dense_19_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_19/BiasAdd/ReadVariableOp?
dense_19/BiasAddBiasAdddense_19/MatMul:product:0'dense_19/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_19/BiasAdds
dense_19/ReluReludense_19/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_19/Relu?
dense_18/MatMul/ReadVariableOpReadVariableOp'dense_18_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02 
dense_18/MatMul/ReadVariableOp?
dense_18/MatMulMatMulflatten_13/Reshape:output:0&dense_18/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_18/MatMul?
dense_18/BiasAdd/ReadVariableOpReadVariableOp(dense_18_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_18/BiasAdd/ReadVariableOp?
dense_18/BiasAddBiasAdddense_18/MatMul:product:0'dense_18/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_18/BiasAdds
dense_18/ReluReludense_18/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_18/Relu?
dense_17/MatMul/ReadVariableOpReadVariableOp'dense_17_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02 
dense_17/MatMul/ReadVariableOp?
dense_17/MatMulMatMulflatten_13/Reshape:output:0&dense_17/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_17/MatMul?
dense_17/BiasAdd/ReadVariableOpReadVariableOp(dense_17_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_17/BiasAdd/ReadVariableOp?
dense_17/BiasAddBiasAdddense_17/MatMul:product:0'dense_17/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_17/BiasAdds
dense_17/ReluReludense_17/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_17/Relu?
dense_16/MatMul/ReadVariableOpReadVariableOp'dense_16_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02 
dense_16/MatMul/ReadVariableOp?
dense_16/MatMulMatMulflatten_13/Reshape:output:0&dense_16/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_16/MatMul?
dense_16/BiasAdd/ReadVariableOpReadVariableOp(dense_16_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_16/BiasAdd/ReadVariableOp?
dense_16/BiasAddBiasAdddense_16/MatMul:product:0'dense_16/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_16/BiasAdds
dense_16/ReluReludense_16/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_16/Relu?
dense_15/MatMul/ReadVariableOpReadVariableOp'dense_15_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02 
dense_15/MatMul/ReadVariableOp?
dense_15/MatMulMatMulflatten_13/Reshape:output:0&dense_15/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_15/MatMul?
dense_15/BiasAdd/ReadVariableOpReadVariableOp(dense_15_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_15/BiasAdd/ReadVariableOp?
dense_15/BiasAddBiasAdddense_15/MatMul:product:0'dense_15/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_15/BiasAdds
dense_15/ReluReludense_15/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_15/Relu?
dense_14/MatMul/ReadVariableOpReadVariableOp'dense_14_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02 
dense_14/MatMul/ReadVariableOp?
dense_14/MatMulMatMulflatten_13/Reshape:output:0&dense_14/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_14/MatMul?
dense_14/BiasAdd/ReadVariableOpReadVariableOp(dense_14_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_14/BiasAdd/ReadVariableOp?
dense_14/BiasAddBiasAdddense_14/MatMul:product:0'dense_14/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_14/BiasAdds
dense_14/ReluReludense_14/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_14/Relu?
dense_13/MatMul/ReadVariableOpReadVariableOp'dense_13_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02 
dense_13/MatMul/ReadVariableOp?
dense_13/MatMulMatMulflatten_13/Reshape:output:0&dense_13/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_13/MatMul?
dense_13/BiasAdd/ReadVariableOpReadVariableOp(dense_13_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_13/BiasAdd/ReadVariableOp?
dense_13/BiasAddBiasAdddense_13/MatMul:product:0'dense_13/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_13/BiasAdds
dense_13/ReluReludense_13/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_13/Relu?
dense_12/MatMul/ReadVariableOpReadVariableOp'dense_12_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02 
dense_12/MatMul/ReadVariableOp?
dense_12/MatMulMatMulflatten_13/Reshape:output:0&dense_12/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_12/MatMul?
dense_12/BiasAdd/ReadVariableOpReadVariableOp(dense_12_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_12/BiasAdd/ReadVariableOp?
dense_12/BiasAddBiasAdddense_12/MatMul:product:0'dense_12/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_12/BiasAdds
dense_12/ReluReludense_12/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_12/Reluu
flatten_25/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
flatten_25/Const?
flatten_25/ReshapeReshapedense_23/Sigmoid:y:0flatten_25/Const:output:0*
T0*'
_output_shapes
:?????????2
flatten_25/Reshapeu
flatten_24/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
flatten_24/Const?
flatten_24/ReshapeReshapedense_22/Relu:activations:0flatten_24/Const:output:0*
T0*'
_output_shapes
:?????????2
flatten_24/Reshapeu
flatten_23/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
flatten_23/Const?
flatten_23/ReshapeReshapedense_21/Relu:activations:0flatten_23/Const:output:0*
T0*'
_output_shapes
:?????????2
flatten_23/Reshapeu
flatten_22/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
flatten_22/Const?
flatten_22/ReshapeReshapedense_20/Relu:activations:0flatten_22/Const:output:0*
T0*'
_output_shapes
:?????????2
flatten_22/Reshapeu
flatten_21/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
flatten_21/Const?
flatten_21/ReshapeReshapedense_19/Relu:activations:0flatten_21/Const:output:0*
T0*'
_output_shapes
:?????????2
flatten_21/Reshapeu
flatten_20/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
flatten_20/Const?
flatten_20/ReshapeReshapedense_18/Relu:activations:0flatten_20/Const:output:0*
T0*'
_output_shapes
:?????????2
flatten_20/Reshapeu
flatten_19/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
flatten_19/Const?
flatten_19/ReshapeReshapedense_17/Relu:activations:0flatten_19/Const:output:0*
T0*'
_output_shapes
:?????????2
flatten_19/Reshapeu
flatten_18/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
flatten_18/Const?
flatten_18/ReshapeReshapedense_16/Relu:activations:0flatten_18/Const:output:0*
T0*'
_output_shapes
:?????????2
flatten_18/Reshapeu
flatten_17/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
flatten_17/Const?
flatten_17/ReshapeReshapedense_15/Relu:activations:0flatten_17/Const:output:0*
T0*'
_output_shapes
:?????????2
flatten_17/Reshapeu
flatten_16/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
flatten_16/Const?
flatten_16/ReshapeReshapedense_14/Relu:activations:0flatten_16/Const:output:0*
T0*'
_output_shapes
:?????????2
flatten_16/Reshapeu
flatten_15/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
flatten_15/Const?
flatten_15/ReshapeReshapedense_13/Relu:activations:0flatten_15/Const:output:0*
T0*'
_output_shapes
:?????????2
flatten_15/Reshapeu
flatten_14/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
flatten_14/Const?
flatten_14/ReshapeReshapedense_12/Relu:activations:0flatten_14/Const:output:0*
T0*'
_output_shapes
:?????????2
flatten_14/Reshape?
IdentityIdentityflatten_14/Reshape:output:0 ^conv3d_3/BiasAdd/ReadVariableOp^conv3d_3/Conv3D/ReadVariableOp ^conv3d_4/BiasAdd/ReadVariableOp^conv3d_4/Conv3D/ReadVariableOp ^dense_12/BiasAdd/ReadVariableOp^dense_12/MatMul/ReadVariableOp ^dense_13/BiasAdd/ReadVariableOp^dense_13/MatMul/ReadVariableOp ^dense_14/BiasAdd/ReadVariableOp^dense_14/MatMul/ReadVariableOp ^dense_15/BiasAdd/ReadVariableOp^dense_15/MatMul/ReadVariableOp ^dense_16/BiasAdd/ReadVariableOp^dense_16/MatMul/ReadVariableOp ^dense_17/BiasAdd/ReadVariableOp^dense_17/MatMul/ReadVariableOp ^dense_18/BiasAdd/ReadVariableOp^dense_18/MatMul/ReadVariableOp ^dense_19/BiasAdd/ReadVariableOp^dense_19/MatMul/ReadVariableOp ^dense_20/BiasAdd/ReadVariableOp^dense_20/MatMul/ReadVariableOp ^dense_21/BiasAdd/ReadVariableOp^dense_21/MatMul/ReadVariableOp ^dense_22/BiasAdd/ReadVariableOp^dense_22/MatMul/ReadVariableOp ^dense_23/BiasAdd/ReadVariableOp^dense_23/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity?

Identity_1Identityflatten_15/Reshape:output:0 ^conv3d_3/BiasAdd/ReadVariableOp^conv3d_3/Conv3D/ReadVariableOp ^conv3d_4/BiasAdd/ReadVariableOp^conv3d_4/Conv3D/ReadVariableOp ^dense_12/BiasAdd/ReadVariableOp^dense_12/MatMul/ReadVariableOp ^dense_13/BiasAdd/ReadVariableOp^dense_13/MatMul/ReadVariableOp ^dense_14/BiasAdd/ReadVariableOp^dense_14/MatMul/ReadVariableOp ^dense_15/BiasAdd/ReadVariableOp^dense_15/MatMul/ReadVariableOp ^dense_16/BiasAdd/ReadVariableOp^dense_16/MatMul/ReadVariableOp ^dense_17/BiasAdd/ReadVariableOp^dense_17/MatMul/ReadVariableOp ^dense_18/BiasAdd/ReadVariableOp^dense_18/MatMul/ReadVariableOp ^dense_19/BiasAdd/ReadVariableOp^dense_19/MatMul/ReadVariableOp ^dense_20/BiasAdd/ReadVariableOp^dense_20/MatMul/ReadVariableOp ^dense_21/BiasAdd/ReadVariableOp^dense_21/MatMul/ReadVariableOp ^dense_22/BiasAdd/ReadVariableOp^dense_22/MatMul/ReadVariableOp ^dense_23/BiasAdd/ReadVariableOp^dense_23/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity_1?

Identity_2Identityflatten_16/Reshape:output:0 ^conv3d_3/BiasAdd/ReadVariableOp^conv3d_3/Conv3D/ReadVariableOp ^conv3d_4/BiasAdd/ReadVariableOp^conv3d_4/Conv3D/ReadVariableOp ^dense_12/BiasAdd/ReadVariableOp^dense_12/MatMul/ReadVariableOp ^dense_13/BiasAdd/ReadVariableOp^dense_13/MatMul/ReadVariableOp ^dense_14/BiasAdd/ReadVariableOp^dense_14/MatMul/ReadVariableOp ^dense_15/BiasAdd/ReadVariableOp^dense_15/MatMul/ReadVariableOp ^dense_16/BiasAdd/ReadVariableOp^dense_16/MatMul/ReadVariableOp ^dense_17/BiasAdd/ReadVariableOp^dense_17/MatMul/ReadVariableOp ^dense_18/BiasAdd/ReadVariableOp^dense_18/MatMul/ReadVariableOp ^dense_19/BiasAdd/ReadVariableOp^dense_19/MatMul/ReadVariableOp ^dense_20/BiasAdd/ReadVariableOp^dense_20/MatMul/ReadVariableOp ^dense_21/BiasAdd/ReadVariableOp^dense_21/MatMul/ReadVariableOp ^dense_22/BiasAdd/ReadVariableOp^dense_22/MatMul/ReadVariableOp ^dense_23/BiasAdd/ReadVariableOp^dense_23/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity_2?

Identity_3Identityflatten_17/Reshape:output:0 ^conv3d_3/BiasAdd/ReadVariableOp^conv3d_3/Conv3D/ReadVariableOp ^conv3d_4/BiasAdd/ReadVariableOp^conv3d_4/Conv3D/ReadVariableOp ^dense_12/BiasAdd/ReadVariableOp^dense_12/MatMul/ReadVariableOp ^dense_13/BiasAdd/ReadVariableOp^dense_13/MatMul/ReadVariableOp ^dense_14/BiasAdd/ReadVariableOp^dense_14/MatMul/ReadVariableOp ^dense_15/BiasAdd/ReadVariableOp^dense_15/MatMul/ReadVariableOp ^dense_16/BiasAdd/ReadVariableOp^dense_16/MatMul/ReadVariableOp ^dense_17/BiasAdd/ReadVariableOp^dense_17/MatMul/ReadVariableOp ^dense_18/BiasAdd/ReadVariableOp^dense_18/MatMul/ReadVariableOp ^dense_19/BiasAdd/ReadVariableOp^dense_19/MatMul/ReadVariableOp ^dense_20/BiasAdd/ReadVariableOp^dense_20/MatMul/ReadVariableOp ^dense_21/BiasAdd/ReadVariableOp^dense_21/MatMul/ReadVariableOp ^dense_22/BiasAdd/ReadVariableOp^dense_22/MatMul/ReadVariableOp ^dense_23/BiasAdd/ReadVariableOp^dense_23/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity_3?

Identity_4Identityflatten_18/Reshape:output:0 ^conv3d_3/BiasAdd/ReadVariableOp^conv3d_3/Conv3D/ReadVariableOp ^conv3d_4/BiasAdd/ReadVariableOp^conv3d_4/Conv3D/ReadVariableOp ^dense_12/BiasAdd/ReadVariableOp^dense_12/MatMul/ReadVariableOp ^dense_13/BiasAdd/ReadVariableOp^dense_13/MatMul/ReadVariableOp ^dense_14/BiasAdd/ReadVariableOp^dense_14/MatMul/ReadVariableOp ^dense_15/BiasAdd/ReadVariableOp^dense_15/MatMul/ReadVariableOp ^dense_16/BiasAdd/ReadVariableOp^dense_16/MatMul/ReadVariableOp ^dense_17/BiasAdd/ReadVariableOp^dense_17/MatMul/ReadVariableOp ^dense_18/BiasAdd/ReadVariableOp^dense_18/MatMul/ReadVariableOp ^dense_19/BiasAdd/ReadVariableOp^dense_19/MatMul/ReadVariableOp ^dense_20/BiasAdd/ReadVariableOp^dense_20/MatMul/ReadVariableOp ^dense_21/BiasAdd/ReadVariableOp^dense_21/MatMul/ReadVariableOp ^dense_22/BiasAdd/ReadVariableOp^dense_22/MatMul/ReadVariableOp ^dense_23/BiasAdd/ReadVariableOp^dense_23/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity_4?

Identity_5Identityflatten_19/Reshape:output:0 ^conv3d_3/BiasAdd/ReadVariableOp^conv3d_3/Conv3D/ReadVariableOp ^conv3d_4/BiasAdd/ReadVariableOp^conv3d_4/Conv3D/ReadVariableOp ^dense_12/BiasAdd/ReadVariableOp^dense_12/MatMul/ReadVariableOp ^dense_13/BiasAdd/ReadVariableOp^dense_13/MatMul/ReadVariableOp ^dense_14/BiasAdd/ReadVariableOp^dense_14/MatMul/ReadVariableOp ^dense_15/BiasAdd/ReadVariableOp^dense_15/MatMul/ReadVariableOp ^dense_16/BiasAdd/ReadVariableOp^dense_16/MatMul/ReadVariableOp ^dense_17/BiasAdd/ReadVariableOp^dense_17/MatMul/ReadVariableOp ^dense_18/BiasAdd/ReadVariableOp^dense_18/MatMul/ReadVariableOp ^dense_19/BiasAdd/ReadVariableOp^dense_19/MatMul/ReadVariableOp ^dense_20/BiasAdd/ReadVariableOp^dense_20/MatMul/ReadVariableOp ^dense_21/BiasAdd/ReadVariableOp^dense_21/MatMul/ReadVariableOp ^dense_22/BiasAdd/ReadVariableOp^dense_22/MatMul/ReadVariableOp ^dense_23/BiasAdd/ReadVariableOp^dense_23/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity_5?

Identity_6Identityflatten_20/Reshape:output:0 ^conv3d_3/BiasAdd/ReadVariableOp^conv3d_3/Conv3D/ReadVariableOp ^conv3d_4/BiasAdd/ReadVariableOp^conv3d_4/Conv3D/ReadVariableOp ^dense_12/BiasAdd/ReadVariableOp^dense_12/MatMul/ReadVariableOp ^dense_13/BiasAdd/ReadVariableOp^dense_13/MatMul/ReadVariableOp ^dense_14/BiasAdd/ReadVariableOp^dense_14/MatMul/ReadVariableOp ^dense_15/BiasAdd/ReadVariableOp^dense_15/MatMul/ReadVariableOp ^dense_16/BiasAdd/ReadVariableOp^dense_16/MatMul/ReadVariableOp ^dense_17/BiasAdd/ReadVariableOp^dense_17/MatMul/ReadVariableOp ^dense_18/BiasAdd/ReadVariableOp^dense_18/MatMul/ReadVariableOp ^dense_19/BiasAdd/ReadVariableOp^dense_19/MatMul/ReadVariableOp ^dense_20/BiasAdd/ReadVariableOp^dense_20/MatMul/ReadVariableOp ^dense_21/BiasAdd/ReadVariableOp^dense_21/MatMul/ReadVariableOp ^dense_22/BiasAdd/ReadVariableOp^dense_22/MatMul/ReadVariableOp ^dense_23/BiasAdd/ReadVariableOp^dense_23/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity_6?

Identity_7Identityflatten_21/Reshape:output:0 ^conv3d_3/BiasAdd/ReadVariableOp^conv3d_3/Conv3D/ReadVariableOp ^conv3d_4/BiasAdd/ReadVariableOp^conv3d_4/Conv3D/ReadVariableOp ^dense_12/BiasAdd/ReadVariableOp^dense_12/MatMul/ReadVariableOp ^dense_13/BiasAdd/ReadVariableOp^dense_13/MatMul/ReadVariableOp ^dense_14/BiasAdd/ReadVariableOp^dense_14/MatMul/ReadVariableOp ^dense_15/BiasAdd/ReadVariableOp^dense_15/MatMul/ReadVariableOp ^dense_16/BiasAdd/ReadVariableOp^dense_16/MatMul/ReadVariableOp ^dense_17/BiasAdd/ReadVariableOp^dense_17/MatMul/ReadVariableOp ^dense_18/BiasAdd/ReadVariableOp^dense_18/MatMul/ReadVariableOp ^dense_19/BiasAdd/ReadVariableOp^dense_19/MatMul/ReadVariableOp ^dense_20/BiasAdd/ReadVariableOp^dense_20/MatMul/ReadVariableOp ^dense_21/BiasAdd/ReadVariableOp^dense_21/MatMul/ReadVariableOp ^dense_22/BiasAdd/ReadVariableOp^dense_22/MatMul/ReadVariableOp ^dense_23/BiasAdd/ReadVariableOp^dense_23/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity_7?

Identity_8Identityflatten_22/Reshape:output:0 ^conv3d_3/BiasAdd/ReadVariableOp^conv3d_3/Conv3D/ReadVariableOp ^conv3d_4/BiasAdd/ReadVariableOp^conv3d_4/Conv3D/ReadVariableOp ^dense_12/BiasAdd/ReadVariableOp^dense_12/MatMul/ReadVariableOp ^dense_13/BiasAdd/ReadVariableOp^dense_13/MatMul/ReadVariableOp ^dense_14/BiasAdd/ReadVariableOp^dense_14/MatMul/ReadVariableOp ^dense_15/BiasAdd/ReadVariableOp^dense_15/MatMul/ReadVariableOp ^dense_16/BiasAdd/ReadVariableOp^dense_16/MatMul/ReadVariableOp ^dense_17/BiasAdd/ReadVariableOp^dense_17/MatMul/ReadVariableOp ^dense_18/BiasAdd/ReadVariableOp^dense_18/MatMul/ReadVariableOp ^dense_19/BiasAdd/ReadVariableOp^dense_19/MatMul/ReadVariableOp ^dense_20/BiasAdd/ReadVariableOp^dense_20/MatMul/ReadVariableOp ^dense_21/BiasAdd/ReadVariableOp^dense_21/MatMul/ReadVariableOp ^dense_22/BiasAdd/ReadVariableOp^dense_22/MatMul/ReadVariableOp ^dense_23/BiasAdd/ReadVariableOp^dense_23/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity_8?

Identity_9Identityflatten_23/Reshape:output:0 ^conv3d_3/BiasAdd/ReadVariableOp^conv3d_3/Conv3D/ReadVariableOp ^conv3d_4/BiasAdd/ReadVariableOp^conv3d_4/Conv3D/ReadVariableOp ^dense_12/BiasAdd/ReadVariableOp^dense_12/MatMul/ReadVariableOp ^dense_13/BiasAdd/ReadVariableOp^dense_13/MatMul/ReadVariableOp ^dense_14/BiasAdd/ReadVariableOp^dense_14/MatMul/ReadVariableOp ^dense_15/BiasAdd/ReadVariableOp^dense_15/MatMul/ReadVariableOp ^dense_16/BiasAdd/ReadVariableOp^dense_16/MatMul/ReadVariableOp ^dense_17/BiasAdd/ReadVariableOp^dense_17/MatMul/ReadVariableOp ^dense_18/BiasAdd/ReadVariableOp^dense_18/MatMul/ReadVariableOp ^dense_19/BiasAdd/ReadVariableOp^dense_19/MatMul/ReadVariableOp ^dense_20/BiasAdd/ReadVariableOp^dense_20/MatMul/ReadVariableOp ^dense_21/BiasAdd/ReadVariableOp^dense_21/MatMul/ReadVariableOp ^dense_22/BiasAdd/ReadVariableOp^dense_22/MatMul/ReadVariableOp ^dense_23/BiasAdd/ReadVariableOp^dense_23/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity_9?
Identity_10Identityflatten_24/Reshape:output:0 ^conv3d_3/BiasAdd/ReadVariableOp^conv3d_3/Conv3D/ReadVariableOp ^conv3d_4/BiasAdd/ReadVariableOp^conv3d_4/Conv3D/ReadVariableOp ^dense_12/BiasAdd/ReadVariableOp^dense_12/MatMul/ReadVariableOp ^dense_13/BiasAdd/ReadVariableOp^dense_13/MatMul/ReadVariableOp ^dense_14/BiasAdd/ReadVariableOp^dense_14/MatMul/ReadVariableOp ^dense_15/BiasAdd/ReadVariableOp^dense_15/MatMul/ReadVariableOp ^dense_16/BiasAdd/ReadVariableOp^dense_16/MatMul/ReadVariableOp ^dense_17/BiasAdd/ReadVariableOp^dense_17/MatMul/ReadVariableOp ^dense_18/BiasAdd/ReadVariableOp^dense_18/MatMul/ReadVariableOp ^dense_19/BiasAdd/ReadVariableOp^dense_19/MatMul/ReadVariableOp ^dense_20/BiasAdd/ReadVariableOp^dense_20/MatMul/ReadVariableOp ^dense_21/BiasAdd/ReadVariableOp^dense_21/MatMul/ReadVariableOp ^dense_22/BiasAdd/ReadVariableOp^dense_22/MatMul/ReadVariableOp ^dense_23/BiasAdd/ReadVariableOp^dense_23/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2
Identity_10?
Identity_11Identityflatten_25/Reshape:output:0 ^conv3d_3/BiasAdd/ReadVariableOp^conv3d_3/Conv3D/ReadVariableOp ^conv3d_4/BiasAdd/ReadVariableOp^conv3d_4/Conv3D/ReadVariableOp ^dense_12/BiasAdd/ReadVariableOp^dense_12/MatMul/ReadVariableOp ^dense_13/BiasAdd/ReadVariableOp^dense_13/MatMul/ReadVariableOp ^dense_14/BiasAdd/ReadVariableOp^dense_14/MatMul/ReadVariableOp ^dense_15/BiasAdd/ReadVariableOp^dense_15/MatMul/ReadVariableOp ^dense_16/BiasAdd/ReadVariableOp^dense_16/MatMul/ReadVariableOp ^dense_17/BiasAdd/ReadVariableOp^dense_17/MatMul/ReadVariableOp ^dense_18/BiasAdd/ReadVariableOp^dense_18/MatMul/ReadVariableOp ^dense_19/BiasAdd/ReadVariableOp^dense_19/MatMul/ReadVariableOp ^dense_20/BiasAdd/ReadVariableOp^dense_20/MatMul/ReadVariableOp ^dense_21/BiasAdd/ReadVariableOp^dense_21/MatMul/ReadVariableOp ^dense_22/BiasAdd/ReadVariableOp^dense_22/MatMul/ReadVariableOp ^dense_23/BiasAdd/ReadVariableOp^dense_23/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2
Identity_11"
identityIdentity:output:0"!

identity_1Identity_1:output:0"#
identity_10Identity_10:output:0"#
identity_11Identity_11:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"!

identity_6Identity_6:output:0"!

identity_7Identity_7:output:0"!

identity_8Identity_8:output:0"!

identity_9Identity_9:output:0*?
_input_shapes?
?:?????????ppd::::::::::::::::::::::::::::2B
conv3d_3/BiasAdd/ReadVariableOpconv3d_3/BiasAdd/ReadVariableOp2@
conv3d_3/Conv3D/ReadVariableOpconv3d_3/Conv3D/ReadVariableOp2B
conv3d_4/BiasAdd/ReadVariableOpconv3d_4/BiasAdd/ReadVariableOp2@
conv3d_4/Conv3D/ReadVariableOpconv3d_4/Conv3D/ReadVariableOp2B
dense_12/BiasAdd/ReadVariableOpdense_12/BiasAdd/ReadVariableOp2@
dense_12/MatMul/ReadVariableOpdense_12/MatMul/ReadVariableOp2B
dense_13/BiasAdd/ReadVariableOpdense_13/BiasAdd/ReadVariableOp2@
dense_13/MatMul/ReadVariableOpdense_13/MatMul/ReadVariableOp2B
dense_14/BiasAdd/ReadVariableOpdense_14/BiasAdd/ReadVariableOp2@
dense_14/MatMul/ReadVariableOpdense_14/MatMul/ReadVariableOp2B
dense_15/BiasAdd/ReadVariableOpdense_15/BiasAdd/ReadVariableOp2@
dense_15/MatMul/ReadVariableOpdense_15/MatMul/ReadVariableOp2B
dense_16/BiasAdd/ReadVariableOpdense_16/BiasAdd/ReadVariableOp2@
dense_16/MatMul/ReadVariableOpdense_16/MatMul/ReadVariableOp2B
dense_17/BiasAdd/ReadVariableOpdense_17/BiasAdd/ReadVariableOp2@
dense_17/MatMul/ReadVariableOpdense_17/MatMul/ReadVariableOp2B
dense_18/BiasAdd/ReadVariableOpdense_18/BiasAdd/ReadVariableOp2@
dense_18/MatMul/ReadVariableOpdense_18/MatMul/ReadVariableOp2B
dense_19/BiasAdd/ReadVariableOpdense_19/BiasAdd/ReadVariableOp2@
dense_19/MatMul/ReadVariableOpdense_19/MatMul/ReadVariableOp2B
dense_20/BiasAdd/ReadVariableOpdense_20/BiasAdd/ReadVariableOp2@
dense_20/MatMul/ReadVariableOpdense_20/MatMul/ReadVariableOp2B
dense_21/BiasAdd/ReadVariableOpdense_21/BiasAdd/ReadVariableOp2@
dense_21/MatMul/ReadVariableOpdense_21/MatMul/ReadVariableOp2B
dense_22/BiasAdd/ReadVariableOpdense_22/BiasAdd/ReadVariableOp2@
dense_22/MatMul/ReadVariableOpdense_22/MatMul/ReadVariableOp2B
dense_23/BiasAdd/ReadVariableOpdense_23/BiasAdd/ReadVariableOp2@
dense_23/MatMul/ReadVariableOpdense_23/MatMul/ReadVariableOp:[ W
3
_output_shapes!
:?????????ppd
 
_user_specified_nameinputs
?
|
'__inference_dense_12_layer_call_fn_9935

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_12_layer_call_and_return_conditional_losses_86602
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
a
E__inference_flatten_22_layer_call_and_return_conditional_losses_10249

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
Constg
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:?????????2	
Reshaped
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
E
)__inference_flatten_13_layer_call_fn_9915

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_flatten_13_layer_call_and_return_conditional_losses_83442
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*2
_input_shapes!
:?????????
:[ W
3
_output_shapes!
:?????????

 
_user_specified_nameinputs
?
e
I__inference_max_pooling3d_3_layer_call_and_return_conditional_losses_8272

inputs
identity?
	MaxPool3D	MaxPool3Dinputs*
T0*W
_output_shapesE
C:A?????????????????????????????????????????????*
ksize	
*
paddingVALID*
strides	
2
	MaxPool3D?
IdentityIdentityMaxPool3D:output:0*
T0*W
_output_shapesE
C:A?????????????????????????????????????????????2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:A?????????????????????????????????????????????: {
W
_output_shapesE
C:A?????????????????????????????????????????????
 
_user_specified_nameinputs
?
`
D__inference_flatten_24_layer_call_and_return_conditional_losses_8696

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
Constg
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:?????????2	
Reshaped
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
`
D__inference_flatten_21_layer_call_and_return_conditional_losses_8738

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
Constg
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:?????????2	
Reshaped
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
??
?

A__inference_model_1_layer_call_and_return_conditional_losses_9242

inputs
conv3d_3_9145
conv3d_3_9147
conv3d_4_9151
conv3d_4_9153
dense_23_9158
dense_23_9160
dense_22_9163
dense_22_9165
dense_21_9168
dense_21_9170
dense_20_9173
dense_20_9175
dense_19_9178
dense_19_9180
dense_18_9183
dense_18_9185
dense_17_9188
dense_17_9190
dense_16_9193
dense_16_9195
dense_15_9198
dense_15_9200
dense_14_9203
dense_14_9205
dense_13_9208
dense_13_9210
dense_12_9213
dense_12_9215
identity

identity_1

identity_2

identity_3

identity_4

identity_5

identity_6

identity_7

identity_8

identity_9
identity_10
identity_11?? conv3d_3/StatefulPartitionedCall? conv3d_4/StatefulPartitionedCall? dense_12/StatefulPartitionedCall? dense_13/StatefulPartitionedCall? dense_14/StatefulPartitionedCall? dense_15/StatefulPartitionedCall? dense_16/StatefulPartitionedCall? dense_17/StatefulPartitionedCall? dense_18/StatefulPartitionedCall? dense_19/StatefulPartitionedCall? dense_20/StatefulPartitionedCall? dense_21/StatefulPartitionedCall? dense_22/StatefulPartitionedCall? dense_23/StatefulPartitionedCall?
 conv3d_3/StatefulPartitionedCallStatefulPartitionedCallinputsconv3d_3_9145conv3d_3_9147*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????66*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_conv3d_3_layer_call_and_return_conditional_losses_82932"
 conv3d_3/StatefulPartitionedCall?
max_pooling3d_2/PartitionedCallPartitionedCall)conv3d_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_max_pooling3d_2_layer_call_and_return_conditional_losses_82602!
max_pooling3d_2/PartitionedCall?
 conv3d_4/StatefulPartitionedCallStatefulPartitionedCall(max_pooling3d_2/PartitionedCall:output:0conv3d_4_9151conv3d_4_9153*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_conv3d_4_layer_call_and_return_conditional_losses_83212"
 conv3d_4/StatefulPartitionedCall?
max_pooling3d_3/PartitionedCallPartitionedCall)conv3d_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_max_pooling3d_3_layer_call_and_return_conditional_losses_82722!
max_pooling3d_3/PartitionedCall?
flatten_13/PartitionedCallPartitionedCall(max_pooling3d_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_flatten_13_layer_call_and_return_conditional_losses_83442
flatten_13/PartitionedCall?
 dense_23/StatefulPartitionedCallStatefulPartitionedCall#flatten_13/PartitionedCall:output:0dense_23_9158dense_23_9160*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_23_layer_call_and_return_conditional_losses_83632"
 dense_23/StatefulPartitionedCall?
 dense_22/StatefulPartitionedCallStatefulPartitionedCall#flatten_13/PartitionedCall:output:0dense_22_9163dense_22_9165*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_22_layer_call_and_return_conditional_losses_83902"
 dense_22/StatefulPartitionedCall?
 dense_21/StatefulPartitionedCallStatefulPartitionedCall#flatten_13/PartitionedCall:output:0dense_21_9168dense_21_9170*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_21_layer_call_and_return_conditional_losses_84172"
 dense_21/StatefulPartitionedCall?
 dense_20/StatefulPartitionedCallStatefulPartitionedCall#flatten_13/PartitionedCall:output:0dense_20_9173dense_20_9175*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_20_layer_call_and_return_conditional_losses_84442"
 dense_20/StatefulPartitionedCall?
 dense_19/StatefulPartitionedCallStatefulPartitionedCall#flatten_13/PartitionedCall:output:0dense_19_9178dense_19_9180*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_19_layer_call_and_return_conditional_losses_84712"
 dense_19/StatefulPartitionedCall?
 dense_18/StatefulPartitionedCallStatefulPartitionedCall#flatten_13/PartitionedCall:output:0dense_18_9183dense_18_9185*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_18_layer_call_and_return_conditional_losses_84982"
 dense_18/StatefulPartitionedCall?
 dense_17/StatefulPartitionedCallStatefulPartitionedCall#flatten_13/PartitionedCall:output:0dense_17_9188dense_17_9190*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_17_layer_call_and_return_conditional_losses_85252"
 dense_17/StatefulPartitionedCall?
 dense_16/StatefulPartitionedCallStatefulPartitionedCall#flatten_13/PartitionedCall:output:0dense_16_9193dense_16_9195*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_16_layer_call_and_return_conditional_losses_85522"
 dense_16/StatefulPartitionedCall?
 dense_15/StatefulPartitionedCallStatefulPartitionedCall#flatten_13/PartitionedCall:output:0dense_15_9198dense_15_9200*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_15_layer_call_and_return_conditional_losses_85792"
 dense_15/StatefulPartitionedCall?
 dense_14/StatefulPartitionedCallStatefulPartitionedCall#flatten_13/PartitionedCall:output:0dense_14_9203dense_14_9205*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_14_layer_call_and_return_conditional_losses_86062"
 dense_14/StatefulPartitionedCall?
 dense_13/StatefulPartitionedCallStatefulPartitionedCall#flatten_13/PartitionedCall:output:0dense_13_9208dense_13_9210*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_13_layer_call_and_return_conditional_losses_86332"
 dense_13/StatefulPartitionedCall?
 dense_12/StatefulPartitionedCallStatefulPartitionedCall#flatten_13/PartitionedCall:output:0dense_12_9213dense_12_9215*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_12_layer_call_and_return_conditional_losses_86602"
 dense_12/StatefulPartitionedCall?
flatten_25/PartitionedCallPartitionedCall)dense_23/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_flatten_25_layer_call_and_return_conditional_losses_86822
flatten_25/PartitionedCall?
flatten_24/PartitionedCallPartitionedCall)dense_22/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_flatten_24_layer_call_and_return_conditional_losses_86962
flatten_24/PartitionedCall?
flatten_23/PartitionedCallPartitionedCall)dense_21/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_flatten_23_layer_call_and_return_conditional_losses_87102
flatten_23/PartitionedCall?
flatten_22/PartitionedCallPartitionedCall)dense_20/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_flatten_22_layer_call_and_return_conditional_losses_87242
flatten_22/PartitionedCall?
flatten_21/PartitionedCallPartitionedCall)dense_19/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_flatten_21_layer_call_and_return_conditional_losses_87382
flatten_21/PartitionedCall?
flatten_20/PartitionedCallPartitionedCall)dense_18/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_flatten_20_layer_call_and_return_conditional_losses_87522
flatten_20/PartitionedCall?
flatten_19/PartitionedCallPartitionedCall)dense_17/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_flatten_19_layer_call_and_return_conditional_losses_87662
flatten_19/PartitionedCall?
flatten_18/PartitionedCallPartitionedCall)dense_16/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_flatten_18_layer_call_and_return_conditional_losses_87802
flatten_18/PartitionedCall?
flatten_17/PartitionedCallPartitionedCall)dense_15/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_flatten_17_layer_call_and_return_conditional_losses_87942
flatten_17/PartitionedCall?
flatten_16/PartitionedCallPartitionedCall)dense_14/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_flatten_16_layer_call_and_return_conditional_losses_88082
flatten_16/PartitionedCall?
flatten_15/PartitionedCallPartitionedCall)dense_13/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_flatten_15_layer_call_and_return_conditional_losses_88222
flatten_15/PartitionedCall?
flatten_14/PartitionedCallPartitionedCall)dense_12/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_flatten_14_layer_call_and_return_conditional_losses_88362
flatten_14/PartitionedCall?
IdentityIdentity#flatten_14/PartitionedCall:output:0!^conv3d_3/StatefulPartitionedCall!^conv3d_4/StatefulPartitionedCall!^dense_12/StatefulPartitionedCall!^dense_13/StatefulPartitionedCall!^dense_14/StatefulPartitionedCall!^dense_15/StatefulPartitionedCall!^dense_16/StatefulPartitionedCall!^dense_17/StatefulPartitionedCall!^dense_18/StatefulPartitionedCall!^dense_19/StatefulPartitionedCall!^dense_20/StatefulPartitionedCall!^dense_21/StatefulPartitionedCall!^dense_22/StatefulPartitionedCall!^dense_23/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity?

Identity_1Identity#flatten_15/PartitionedCall:output:0!^conv3d_3/StatefulPartitionedCall!^conv3d_4/StatefulPartitionedCall!^dense_12/StatefulPartitionedCall!^dense_13/StatefulPartitionedCall!^dense_14/StatefulPartitionedCall!^dense_15/StatefulPartitionedCall!^dense_16/StatefulPartitionedCall!^dense_17/StatefulPartitionedCall!^dense_18/StatefulPartitionedCall!^dense_19/StatefulPartitionedCall!^dense_20/StatefulPartitionedCall!^dense_21/StatefulPartitionedCall!^dense_22/StatefulPartitionedCall!^dense_23/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_1?

Identity_2Identity#flatten_16/PartitionedCall:output:0!^conv3d_3/StatefulPartitionedCall!^conv3d_4/StatefulPartitionedCall!^dense_12/StatefulPartitionedCall!^dense_13/StatefulPartitionedCall!^dense_14/StatefulPartitionedCall!^dense_15/StatefulPartitionedCall!^dense_16/StatefulPartitionedCall!^dense_17/StatefulPartitionedCall!^dense_18/StatefulPartitionedCall!^dense_19/StatefulPartitionedCall!^dense_20/StatefulPartitionedCall!^dense_21/StatefulPartitionedCall!^dense_22/StatefulPartitionedCall!^dense_23/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_2?

Identity_3Identity#flatten_17/PartitionedCall:output:0!^conv3d_3/StatefulPartitionedCall!^conv3d_4/StatefulPartitionedCall!^dense_12/StatefulPartitionedCall!^dense_13/StatefulPartitionedCall!^dense_14/StatefulPartitionedCall!^dense_15/StatefulPartitionedCall!^dense_16/StatefulPartitionedCall!^dense_17/StatefulPartitionedCall!^dense_18/StatefulPartitionedCall!^dense_19/StatefulPartitionedCall!^dense_20/StatefulPartitionedCall!^dense_21/StatefulPartitionedCall!^dense_22/StatefulPartitionedCall!^dense_23/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_3?

Identity_4Identity#flatten_18/PartitionedCall:output:0!^conv3d_3/StatefulPartitionedCall!^conv3d_4/StatefulPartitionedCall!^dense_12/StatefulPartitionedCall!^dense_13/StatefulPartitionedCall!^dense_14/StatefulPartitionedCall!^dense_15/StatefulPartitionedCall!^dense_16/StatefulPartitionedCall!^dense_17/StatefulPartitionedCall!^dense_18/StatefulPartitionedCall!^dense_19/StatefulPartitionedCall!^dense_20/StatefulPartitionedCall!^dense_21/StatefulPartitionedCall!^dense_22/StatefulPartitionedCall!^dense_23/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_4?

Identity_5Identity#flatten_19/PartitionedCall:output:0!^conv3d_3/StatefulPartitionedCall!^conv3d_4/StatefulPartitionedCall!^dense_12/StatefulPartitionedCall!^dense_13/StatefulPartitionedCall!^dense_14/StatefulPartitionedCall!^dense_15/StatefulPartitionedCall!^dense_16/StatefulPartitionedCall!^dense_17/StatefulPartitionedCall!^dense_18/StatefulPartitionedCall!^dense_19/StatefulPartitionedCall!^dense_20/StatefulPartitionedCall!^dense_21/StatefulPartitionedCall!^dense_22/StatefulPartitionedCall!^dense_23/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_5?

Identity_6Identity#flatten_20/PartitionedCall:output:0!^conv3d_3/StatefulPartitionedCall!^conv3d_4/StatefulPartitionedCall!^dense_12/StatefulPartitionedCall!^dense_13/StatefulPartitionedCall!^dense_14/StatefulPartitionedCall!^dense_15/StatefulPartitionedCall!^dense_16/StatefulPartitionedCall!^dense_17/StatefulPartitionedCall!^dense_18/StatefulPartitionedCall!^dense_19/StatefulPartitionedCall!^dense_20/StatefulPartitionedCall!^dense_21/StatefulPartitionedCall!^dense_22/StatefulPartitionedCall!^dense_23/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_6?

Identity_7Identity#flatten_21/PartitionedCall:output:0!^conv3d_3/StatefulPartitionedCall!^conv3d_4/StatefulPartitionedCall!^dense_12/StatefulPartitionedCall!^dense_13/StatefulPartitionedCall!^dense_14/StatefulPartitionedCall!^dense_15/StatefulPartitionedCall!^dense_16/StatefulPartitionedCall!^dense_17/StatefulPartitionedCall!^dense_18/StatefulPartitionedCall!^dense_19/StatefulPartitionedCall!^dense_20/StatefulPartitionedCall!^dense_21/StatefulPartitionedCall!^dense_22/StatefulPartitionedCall!^dense_23/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_7?

Identity_8Identity#flatten_22/PartitionedCall:output:0!^conv3d_3/StatefulPartitionedCall!^conv3d_4/StatefulPartitionedCall!^dense_12/StatefulPartitionedCall!^dense_13/StatefulPartitionedCall!^dense_14/StatefulPartitionedCall!^dense_15/StatefulPartitionedCall!^dense_16/StatefulPartitionedCall!^dense_17/StatefulPartitionedCall!^dense_18/StatefulPartitionedCall!^dense_19/StatefulPartitionedCall!^dense_20/StatefulPartitionedCall!^dense_21/StatefulPartitionedCall!^dense_22/StatefulPartitionedCall!^dense_23/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_8?

Identity_9Identity#flatten_23/PartitionedCall:output:0!^conv3d_3/StatefulPartitionedCall!^conv3d_4/StatefulPartitionedCall!^dense_12/StatefulPartitionedCall!^dense_13/StatefulPartitionedCall!^dense_14/StatefulPartitionedCall!^dense_15/StatefulPartitionedCall!^dense_16/StatefulPartitionedCall!^dense_17/StatefulPartitionedCall!^dense_18/StatefulPartitionedCall!^dense_19/StatefulPartitionedCall!^dense_20/StatefulPartitionedCall!^dense_21/StatefulPartitionedCall!^dense_22/StatefulPartitionedCall!^dense_23/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_9?
Identity_10Identity#flatten_24/PartitionedCall:output:0!^conv3d_3/StatefulPartitionedCall!^conv3d_4/StatefulPartitionedCall!^dense_12/StatefulPartitionedCall!^dense_13/StatefulPartitionedCall!^dense_14/StatefulPartitionedCall!^dense_15/StatefulPartitionedCall!^dense_16/StatefulPartitionedCall!^dense_17/StatefulPartitionedCall!^dense_18/StatefulPartitionedCall!^dense_19/StatefulPartitionedCall!^dense_20/StatefulPartitionedCall!^dense_21/StatefulPartitionedCall!^dense_22/StatefulPartitionedCall!^dense_23/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2
Identity_10?
Identity_11Identity#flatten_25/PartitionedCall:output:0!^conv3d_3/StatefulPartitionedCall!^conv3d_4/StatefulPartitionedCall!^dense_12/StatefulPartitionedCall!^dense_13/StatefulPartitionedCall!^dense_14/StatefulPartitionedCall!^dense_15/StatefulPartitionedCall!^dense_16/StatefulPartitionedCall!^dense_17/StatefulPartitionedCall!^dense_18/StatefulPartitionedCall!^dense_19/StatefulPartitionedCall!^dense_20/StatefulPartitionedCall!^dense_21/StatefulPartitionedCall!^dense_22/StatefulPartitionedCall!^dense_23/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2
Identity_11"
identityIdentity:output:0"!

identity_1Identity_1:output:0"#
identity_10Identity_10:output:0"#
identity_11Identity_11:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"!

identity_6Identity_6:output:0"!

identity_7Identity_7:output:0"!

identity_8Identity_8:output:0"!

identity_9Identity_9:output:0*?
_input_shapes?
?:?????????ppd::::::::::::::::::::::::::::2D
 conv3d_3/StatefulPartitionedCall conv3d_3/StatefulPartitionedCall2D
 conv3d_4/StatefulPartitionedCall conv3d_4/StatefulPartitionedCall2D
 dense_12/StatefulPartitionedCall dense_12/StatefulPartitionedCall2D
 dense_13/StatefulPartitionedCall dense_13/StatefulPartitionedCall2D
 dense_14/StatefulPartitionedCall dense_14/StatefulPartitionedCall2D
 dense_15/StatefulPartitionedCall dense_15/StatefulPartitionedCall2D
 dense_16/StatefulPartitionedCall dense_16/StatefulPartitionedCall2D
 dense_17/StatefulPartitionedCall dense_17/StatefulPartitionedCall2D
 dense_18/StatefulPartitionedCall dense_18/StatefulPartitionedCall2D
 dense_19/StatefulPartitionedCall dense_19/StatefulPartitionedCall2D
 dense_20/StatefulPartitionedCall dense_20/StatefulPartitionedCall2D
 dense_21/StatefulPartitionedCall dense_21/StatefulPartitionedCall2D
 dense_22/StatefulPartitionedCall dense_22/StatefulPartitionedCall2D
 dense_23/StatefulPartitionedCall dense_23/StatefulPartitionedCall:[ W
3
_output_shapes!
:?????????ppd
 
_user_specified_nameinputs
?	
?
B__inference_dense_13_layer_call_and_return_conditional_losses_8633

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
??
?
__inference__wrapped_model_8254
input_23
/model_1_conv3d_3_conv3d_readvariableop_resource4
0model_1_conv3d_3_biasadd_readvariableop_resource3
/model_1_conv3d_4_conv3d_readvariableop_resource4
0model_1_conv3d_4_biasadd_readvariableop_resource3
/model_1_dense_23_matmul_readvariableop_resource4
0model_1_dense_23_biasadd_readvariableop_resource3
/model_1_dense_22_matmul_readvariableop_resource4
0model_1_dense_22_biasadd_readvariableop_resource3
/model_1_dense_21_matmul_readvariableop_resource4
0model_1_dense_21_biasadd_readvariableop_resource3
/model_1_dense_20_matmul_readvariableop_resource4
0model_1_dense_20_biasadd_readvariableop_resource3
/model_1_dense_19_matmul_readvariableop_resource4
0model_1_dense_19_biasadd_readvariableop_resource3
/model_1_dense_18_matmul_readvariableop_resource4
0model_1_dense_18_biasadd_readvariableop_resource3
/model_1_dense_17_matmul_readvariableop_resource4
0model_1_dense_17_biasadd_readvariableop_resource3
/model_1_dense_16_matmul_readvariableop_resource4
0model_1_dense_16_biasadd_readvariableop_resource3
/model_1_dense_15_matmul_readvariableop_resource4
0model_1_dense_15_biasadd_readvariableop_resource3
/model_1_dense_14_matmul_readvariableop_resource4
0model_1_dense_14_biasadd_readvariableop_resource3
/model_1_dense_13_matmul_readvariableop_resource4
0model_1_dense_13_biasadd_readvariableop_resource3
/model_1_dense_12_matmul_readvariableop_resource4
0model_1_dense_12_biasadd_readvariableop_resource
identity

identity_1

identity_2

identity_3

identity_4

identity_5

identity_6

identity_7

identity_8

identity_9
identity_10
identity_11??'model_1/conv3d_3/BiasAdd/ReadVariableOp?&model_1/conv3d_3/Conv3D/ReadVariableOp?'model_1/conv3d_4/BiasAdd/ReadVariableOp?&model_1/conv3d_4/Conv3D/ReadVariableOp?'model_1/dense_12/BiasAdd/ReadVariableOp?&model_1/dense_12/MatMul/ReadVariableOp?'model_1/dense_13/BiasAdd/ReadVariableOp?&model_1/dense_13/MatMul/ReadVariableOp?'model_1/dense_14/BiasAdd/ReadVariableOp?&model_1/dense_14/MatMul/ReadVariableOp?'model_1/dense_15/BiasAdd/ReadVariableOp?&model_1/dense_15/MatMul/ReadVariableOp?'model_1/dense_16/BiasAdd/ReadVariableOp?&model_1/dense_16/MatMul/ReadVariableOp?'model_1/dense_17/BiasAdd/ReadVariableOp?&model_1/dense_17/MatMul/ReadVariableOp?'model_1/dense_18/BiasAdd/ReadVariableOp?&model_1/dense_18/MatMul/ReadVariableOp?'model_1/dense_19/BiasAdd/ReadVariableOp?&model_1/dense_19/MatMul/ReadVariableOp?'model_1/dense_20/BiasAdd/ReadVariableOp?&model_1/dense_20/MatMul/ReadVariableOp?'model_1/dense_21/BiasAdd/ReadVariableOp?&model_1/dense_21/MatMul/ReadVariableOp?'model_1/dense_22/BiasAdd/ReadVariableOp?&model_1/dense_22/MatMul/ReadVariableOp?'model_1/dense_23/BiasAdd/ReadVariableOp?&model_1/dense_23/MatMul/ReadVariableOp?
&model_1/conv3d_3/Conv3D/ReadVariableOpReadVariableOp/model_1_conv3d_3_conv3d_readvariableop_resource**
_output_shapes
:d*
dtype02(
&model_1/conv3d_3/Conv3D/ReadVariableOp?
model_1/conv3d_3/Conv3DConv3Dinput_2.model_1/conv3d_3/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????66*
paddingVALID*
strides	
2
model_1/conv3d_3/Conv3D?
'model_1/conv3d_3/BiasAdd/ReadVariableOpReadVariableOp0model_1_conv3d_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02)
'model_1/conv3d_3/BiasAdd/ReadVariableOp?
model_1/conv3d_3/BiasAddBiasAdd model_1/conv3d_3/Conv3D:output:0/model_1/conv3d_3/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????662
model_1/conv3d_3/BiasAdd?
model_1/conv3d_3/ReluRelu!model_1/conv3d_3/BiasAdd:output:0*
T0*3
_output_shapes!
:?????????662
model_1/conv3d_3/Relu?
!model_1/max_pooling3d_2/MaxPool3D	MaxPool3D#model_1/conv3d_3/Relu:activations:0*
T0*3
_output_shapes!
:?????????*
ksize	
*
paddingVALID*
strides	
2#
!model_1/max_pooling3d_2/MaxPool3D?
&model_1/conv3d_4/Conv3D/ReadVariableOpReadVariableOp/model_1_conv3d_4_conv3d_readvariableop_resource**
_output_shapes
:
*
dtype02(
&model_1/conv3d_4/Conv3D/ReadVariableOp?
model_1/conv3d_4/Conv3DConv3D*model_1/max_pooling3d_2/MaxPool3D:output:0.model_1/conv3d_4/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????
*
paddingVALID*
strides	
2
model_1/conv3d_4/Conv3D?
'model_1/conv3d_4/BiasAdd/ReadVariableOpReadVariableOp0model_1_conv3d_4_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02)
'model_1/conv3d_4/BiasAdd/ReadVariableOp?
model_1/conv3d_4/BiasAddBiasAdd model_1/conv3d_4/Conv3D:output:0/model_1/conv3d_4/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????
2
model_1/conv3d_4/BiasAdd?
model_1/conv3d_4/ReluRelu!model_1/conv3d_4/BiasAdd:output:0*
T0*3
_output_shapes!
:?????????
2
model_1/conv3d_4/Relu?
!model_1/max_pooling3d_3/MaxPool3D	MaxPool3D#model_1/conv3d_4/Relu:activations:0*
T0*3
_output_shapes!
:?????????
*
ksize	
*
paddingVALID*
strides	
2#
!model_1/max_pooling3d_3/MaxPool3D?
model_1/flatten_13/ConstConst*
_output_shapes
:*
dtype0*
valueB"????h  2
model_1/flatten_13/Const?
model_1/flatten_13/ReshapeReshape*model_1/max_pooling3d_3/MaxPool3D:output:0!model_1/flatten_13/Const:output:0*
T0*(
_output_shapes
:??????????2
model_1/flatten_13/Reshape?
&model_1/dense_23/MatMul/ReadVariableOpReadVariableOp/model_1_dense_23_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02(
&model_1/dense_23/MatMul/ReadVariableOp?
model_1/dense_23/MatMulMatMul#model_1/flatten_13/Reshape:output:0.model_1/dense_23/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
model_1/dense_23/MatMul?
'model_1/dense_23/BiasAdd/ReadVariableOpReadVariableOp0model_1_dense_23_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02)
'model_1/dense_23/BiasAdd/ReadVariableOp?
model_1/dense_23/BiasAddBiasAdd!model_1/dense_23/MatMul:product:0/model_1/dense_23/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
model_1/dense_23/BiasAdd?
model_1/dense_23/SigmoidSigmoid!model_1/dense_23/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
model_1/dense_23/Sigmoid?
&model_1/dense_22/MatMul/ReadVariableOpReadVariableOp/model_1_dense_22_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02(
&model_1/dense_22/MatMul/ReadVariableOp?
model_1/dense_22/MatMulMatMul#model_1/flatten_13/Reshape:output:0.model_1/dense_22/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
model_1/dense_22/MatMul?
'model_1/dense_22/BiasAdd/ReadVariableOpReadVariableOp0model_1_dense_22_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02)
'model_1/dense_22/BiasAdd/ReadVariableOp?
model_1/dense_22/BiasAddBiasAdd!model_1/dense_22/MatMul:product:0/model_1/dense_22/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
model_1/dense_22/BiasAdd?
model_1/dense_22/ReluRelu!model_1/dense_22/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
model_1/dense_22/Relu?
&model_1/dense_21/MatMul/ReadVariableOpReadVariableOp/model_1_dense_21_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02(
&model_1/dense_21/MatMul/ReadVariableOp?
model_1/dense_21/MatMulMatMul#model_1/flatten_13/Reshape:output:0.model_1/dense_21/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
model_1/dense_21/MatMul?
'model_1/dense_21/BiasAdd/ReadVariableOpReadVariableOp0model_1_dense_21_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02)
'model_1/dense_21/BiasAdd/ReadVariableOp?
model_1/dense_21/BiasAddBiasAdd!model_1/dense_21/MatMul:product:0/model_1/dense_21/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
model_1/dense_21/BiasAdd?
model_1/dense_21/ReluRelu!model_1/dense_21/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
model_1/dense_21/Relu?
&model_1/dense_20/MatMul/ReadVariableOpReadVariableOp/model_1_dense_20_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02(
&model_1/dense_20/MatMul/ReadVariableOp?
model_1/dense_20/MatMulMatMul#model_1/flatten_13/Reshape:output:0.model_1/dense_20/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
model_1/dense_20/MatMul?
'model_1/dense_20/BiasAdd/ReadVariableOpReadVariableOp0model_1_dense_20_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02)
'model_1/dense_20/BiasAdd/ReadVariableOp?
model_1/dense_20/BiasAddBiasAdd!model_1/dense_20/MatMul:product:0/model_1/dense_20/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
model_1/dense_20/BiasAdd?
model_1/dense_20/ReluRelu!model_1/dense_20/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
model_1/dense_20/Relu?
&model_1/dense_19/MatMul/ReadVariableOpReadVariableOp/model_1_dense_19_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02(
&model_1/dense_19/MatMul/ReadVariableOp?
model_1/dense_19/MatMulMatMul#model_1/flatten_13/Reshape:output:0.model_1/dense_19/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
model_1/dense_19/MatMul?
'model_1/dense_19/BiasAdd/ReadVariableOpReadVariableOp0model_1_dense_19_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02)
'model_1/dense_19/BiasAdd/ReadVariableOp?
model_1/dense_19/BiasAddBiasAdd!model_1/dense_19/MatMul:product:0/model_1/dense_19/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
model_1/dense_19/BiasAdd?
model_1/dense_19/ReluRelu!model_1/dense_19/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
model_1/dense_19/Relu?
&model_1/dense_18/MatMul/ReadVariableOpReadVariableOp/model_1_dense_18_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02(
&model_1/dense_18/MatMul/ReadVariableOp?
model_1/dense_18/MatMulMatMul#model_1/flatten_13/Reshape:output:0.model_1/dense_18/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
model_1/dense_18/MatMul?
'model_1/dense_18/BiasAdd/ReadVariableOpReadVariableOp0model_1_dense_18_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02)
'model_1/dense_18/BiasAdd/ReadVariableOp?
model_1/dense_18/BiasAddBiasAdd!model_1/dense_18/MatMul:product:0/model_1/dense_18/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
model_1/dense_18/BiasAdd?
model_1/dense_18/ReluRelu!model_1/dense_18/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
model_1/dense_18/Relu?
&model_1/dense_17/MatMul/ReadVariableOpReadVariableOp/model_1_dense_17_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02(
&model_1/dense_17/MatMul/ReadVariableOp?
model_1/dense_17/MatMulMatMul#model_1/flatten_13/Reshape:output:0.model_1/dense_17/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
model_1/dense_17/MatMul?
'model_1/dense_17/BiasAdd/ReadVariableOpReadVariableOp0model_1_dense_17_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02)
'model_1/dense_17/BiasAdd/ReadVariableOp?
model_1/dense_17/BiasAddBiasAdd!model_1/dense_17/MatMul:product:0/model_1/dense_17/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
model_1/dense_17/BiasAdd?
model_1/dense_17/ReluRelu!model_1/dense_17/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
model_1/dense_17/Relu?
&model_1/dense_16/MatMul/ReadVariableOpReadVariableOp/model_1_dense_16_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02(
&model_1/dense_16/MatMul/ReadVariableOp?
model_1/dense_16/MatMulMatMul#model_1/flatten_13/Reshape:output:0.model_1/dense_16/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
model_1/dense_16/MatMul?
'model_1/dense_16/BiasAdd/ReadVariableOpReadVariableOp0model_1_dense_16_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02)
'model_1/dense_16/BiasAdd/ReadVariableOp?
model_1/dense_16/BiasAddBiasAdd!model_1/dense_16/MatMul:product:0/model_1/dense_16/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
model_1/dense_16/BiasAdd?
model_1/dense_16/ReluRelu!model_1/dense_16/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
model_1/dense_16/Relu?
&model_1/dense_15/MatMul/ReadVariableOpReadVariableOp/model_1_dense_15_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02(
&model_1/dense_15/MatMul/ReadVariableOp?
model_1/dense_15/MatMulMatMul#model_1/flatten_13/Reshape:output:0.model_1/dense_15/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
model_1/dense_15/MatMul?
'model_1/dense_15/BiasAdd/ReadVariableOpReadVariableOp0model_1_dense_15_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02)
'model_1/dense_15/BiasAdd/ReadVariableOp?
model_1/dense_15/BiasAddBiasAdd!model_1/dense_15/MatMul:product:0/model_1/dense_15/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
model_1/dense_15/BiasAdd?
model_1/dense_15/ReluRelu!model_1/dense_15/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
model_1/dense_15/Relu?
&model_1/dense_14/MatMul/ReadVariableOpReadVariableOp/model_1_dense_14_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02(
&model_1/dense_14/MatMul/ReadVariableOp?
model_1/dense_14/MatMulMatMul#model_1/flatten_13/Reshape:output:0.model_1/dense_14/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
model_1/dense_14/MatMul?
'model_1/dense_14/BiasAdd/ReadVariableOpReadVariableOp0model_1_dense_14_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02)
'model_1/dense_14/BiasAdd/ReadVariableOp?
model_1/dense_14/BiasAddBiasAdd!model_1/dense_14/MatMul:product:0/model_1/dense_14/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
model_1/dense_14/BiasAdd?
model_1/dense_14/ReluRelu!model_1/dense_14/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
model_1/dense_14/Relu?
&model_1/dense_13/MatMul/ReadVariableOpReadVariableOp/model_1_dense_13_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02(
&model_1/dense_13/MatMul/ReadVariableOp?
model_1/dense_13/MatMulMatMul#model_1/flatten_13/Reshape:output:0.model_1/dense_13/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
model_1/dense_13/MatMul?
'model_1/dense_13/BiasAdd/ReadVariableOpReadVariableOp0model_1_dense_13_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02)
'model_1/dense_13/BiasAdd/ReadVariableOp?
model_1/dense_13/BiasAddBiasAdd!model_1/dense_13/MatMul:product:0/model_1/dense_13/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
model_1/dense_13/BiasAdd?
model_1/dense_13/ReluRelu!model_1/dense_13/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
model_1/dense_13/Relu?
&model_1/dense_12/MatMul/ReadVariableOpReadVariableOp/model_1_dense_12_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02(
&model_1/dense_12/MatMul/ReadVariableOp?
model_1/dense_12/MatMulMatMul#model_1/flatten_13/Reshape:output:0.model_1/dense_12/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
model_1/dense_12/MatMul?
'model_1/dense_12/BiasAdd/ReadVariableOpReadVariableOp0model_1_dense_12_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02)
'model_1/dense_12/BiasAdd/ReadVariableOp?
model_1/dense_12/BiasAddBiasAdd!model_1/dense_12/MatMul:product:0/model_1/dense_12/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
model_1/dense_12/BiasAdd?
model_1/dense_12/ReluRelu!model_1/dense_12/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
model_1/dense_12/Relu?
model_1/flatten_25/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
model_1/flatten_25/Const?
model_1/flatten_25/ReshapeReshapemodel_1/dense_23/Sigmoid:y:0!model_1/flatten_25/Const:output:0*
T0*'
_output_shapes
:?????????2
model_1/flatten_25/Reshape?
model_1/flatten_24/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
model_1/flatten_24/Const?
model_1/flatten_24/ReshapeReshape#model_1/dense_22/Relu:activations:0!model_1/flatten_24/Const:output:0*
T0*'
_output_shapes
:?????????2
model_1/flatten_24/Reshape?
model_1/flatten_23/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
model_1/flatten_23/Const?
model_1/flatten_23/ReshapeReshape#model_1/dense_21/Relu:activations:0!model_1/flatten_23/Const:output:0*
T0*'
_output_shapes
:?????????2
model_1/flatten_23/Reshape?
model_1/flatten_22/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
model_1/flatten_22/Const?
model_1/flatten_22/ReshapeReshape#model_1/dense_20/Relu:activations:0!model_1/flatten_22/Const:output:0*
T0*'
_output_shapes
:?????????2
model_1/flatten_22/Reshape?
model_1/flatten_21/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
model_1/flatten_21/Const?
model_1/flatten_21/ReshapeReshape#model_1/dense_19/Relu:activations:0!model_1/flatten_21/Const:output:0*
T0*'
_output_shapes
:?????????2
model_1/flatten_21/Reshape?
model_1/flatten_20/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
model_1/flatten_20/Const?
model_1/flatten_20/ReshapeReshape#model_1/dense_18/Relu:activations:0!model_1/flatten_20/Const:output:0*
T0*'
_output_shapes
:?????????2
model_1/flatten_20/Reshape?
model_1/flatten_19/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
model_1/flatten_19/Const?
model_1/flatten_19/ReshapeReshape#model_1/dense_17/Relu:activations:0!model_1/flatten_19/Const:output:0*
T0*'
_output_shapes
:?????????2
model_1/flatten_19/Reshape?
model_1/flatten_18/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
model_1/flatten_18/Const?
model_1/flatten_18/ReshapeReshape#model_1/dense_16/Relu:activations:0!model_1/flatten_18/Const:output:0*
T0*'
_output_shapes
:?????????2
model_1/flatten_18/Reshape?
model_1/flatten_17/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
model_1/flatten_17/Const?
model_1/flatten_17/ReshapeReshape#model_1/dense_15/Relu:activations:0!model_1/flatten_17/Const:output:0*
T0*'
_output_shapes
:?????????2
model_1/flatten_17/Reshape?
model_1/flatten_16/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
model_1/flatten_16/Const?
model_1/flatten_16/ReshapeReshape#model_1/dense_14/Relu:activations:0!model_1/flatten_16/Const:output:0*
T0*'
_output_shapes
:?????????2
model_1/flatten_16/Reshape?
model_1/flatten_15/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
model_1/flatten_15/Const?
model_1/flatten_15/ReshapeReshape#model_1/dense_13/Relu:activations:0!model_1/flatten_15/Const:output:0*
T0*'
_output_shapes
:?????????2
model_1/flatten_15/Reshape?
model_1/flatten_14/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
model_1/flatten_14/Const?
model_1/flatten_14/ReshapeReshape#model_1/dense_12/Relu:activations:0!model_1/flatten_14/Const:output:0*
T0*'
_output_shapes
:?????????2
model_1/flatten_14/Reshape?

IdentityIdentity#model_1/flatten_14/Reshape:output:0(^model_1/conv3d_3/BiasAdd/ReadVariableOp'^model_1/conv3d_3/Conv3D/ReadVariableOp(^model_1/conv3d_4/BiasAdd/ReadVariableOp'^model_1/conv3d_4/Conv3D/ReadVariableOp(^model_1/dense_12/BiasAdd/ReadVariableOp'^model_1/dense_12/MatMul/ReadVariableOp(^model_1/dense_13/BiasAdd/ReadVariableOp'^model_1/dense_13/MatMul/ReadVariableOp(^model_1/dense_14/BiasAdd/ReadVariableOp'^model_1/dense_14/MatMul/ReadVariableOp(^model_1/dense_15/BiasAdd/ReadVariableOp'^model_1/dense_15/MatMul/ReadVariableOp(^model_1/dense_16/BiasAdd/ReadVariableOp'^model_1/dense_16/MatMul/ReadVariableOp(^model_1/dense_17/BiasAdd/ReadVariableOp'^model_1/dense_17/MatMul/ReadVariableOp(^model_1/dense_18/BiasAdd/ReadVariableOp'^model_1/dense_18/MatMul/ReadVariableOp(^model_1/dense_19/BiasAdd/ReadVariableOp'^model_1/dense_19/MatMul/ReadVariableOp(^model_1/dense_20/BiasAdd/ReadVariableOp'^model_1/dense_20/MatMul/ReadVariableOp(^model_1/dense_21/BiasAdd/ReadVariableOp'^model_1/dense_21/MatMul/ReadVariableOp(^model_1/dense_22/BiasAdd/ReadVariableOp'^model_1/dense_22/MatMul/ReadVariableOp(^model_1/dense_23/BiasAdd/ReadVariableOp'^model_1/dense_23/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity?


Identity_1Identity#model_1/flatten_15/Reshape:output:0(^model_1/conv3d_3/BiasAdd/ReadVariableOp'^model_1/conv3d_3/Conv3D/ReadVariableOp(^model_1/conv3d_4/BiasAdd/ReadVariableOp'^model_1/conv3d_4/Conv3D/ReadVariableOp(^model_1/dense_12/BiasAdd/ReadVariableOp'^model_1/dense_12/MatMul/ReadVariableOp(^model_1/dense_13/BiasAdd/ReadVariableOp'^model_1/dense_13/MatMul/ReadVariableOp(^model_1/dense_14/BiasAdd/ReadVariableOp'^model_1/dense_14/MatMul/ReadVariableOp(^model_1/dense_15/BiasAdd/ReadVariableOp'^model_1/dense_15/MatMul/ReadVariableOp(^model_1/dense_16/BiasAdd/ReadVariableOp'^model_1/dense_16/MatMul/ReadVariableOp(^model_1/dense_17/BiasAdd/ReadVariableOp'^model_1/dense_17/MatMul/ReadVariableOp(^model_1/dense_18/BiasAdd/ReadVariableOp'^model_1/dense_18/MatMul/ReadVariableOp(^model_1/dense_19/BiasAdd/ReadVariableOp'^model_1/dense_19/MatMul/ReadVariableOp(^model_1/dense_20/BiasAdd/ReadVariableOp'^model_1/dense_20/MatMul/ReadVariableOp(^model_1/dense_21/BiasAdd/ReadVariableOp'^model_1/dense_21/MatMul/ReadVariableOp(^model_1/dense_22/BiasAdd/ReadVariableOp'^model_1/dense_22/MatMul/ReadVariableOp(^model_1/dense_23/BiasAdd/ReadVariableOp'^model_1/dense_23/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity_1?


Identity_2Identity#model_1/flatten_16/Reshape:output:0(^model_1/conv3d_3/BiasAdd/ReadVariableOp'^model_1/conv3d_3/Conv3D/ReadVariableOp(^model_1/conv3d_4/BiasAdd/ReadVariableOp'^model_1/conv3d_4/Conv3D/ReadVariableOp(^model_1/dense_12/BiasAdd/ReadVariableOp'^model_1/dense_12/MatMul/ReadVariableOp(^model_1/dense_13/BiasAdd/ReadVariableOp'^model_1/dense_13/MatMul/ReadVariableOp(^model_1/dense_14/BiasAdd/ReadVariableOp'^model_1/dense_14/MatMul/ReadVariableOp(^model_1/dense_15/BiasAdd/ReadVariableOp'^model_1/dense_15/MatMul/ReadVariableOp(^model_1/dense_16/BiasAdd/ReadVariableOp'^model_1/dense_16/MatMul/ReadVariableOp(^model_1/dense_17/BiasAdd/ReadVariableOp'^model_1/dense_17/MatMul/ReadVariableOp(^model_1/dense_18/BiasAdd/ReadVariableOp'^model_1/dense_18/MatMul/ReadVariableOp(^model_1/dense_19/BiasAdd/ReadVariableOp'^model_1/dense_19/MatMul/ReadVariableOp(^model_1/dense_20/BiasAdd/ReadVariableOp'^model_1/dense_20/MatMul/ReadVariableOp(^model_1/dense_21/BiasAdd/ReadVariableOp'^model_1/dense_21/MatMul/ReadVariableOp(^model_1/dense_22/BiasAdd/ReadVariableOp'^model_1/dense_22/MatMul/ReadVariableOp(^model_1/dense_23/BiasAdd/ReadVariableOp'^model_1/dense_23/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity_2?


Identity_3Identity#model_1/flatten_17/Reshape:output:0(^model_1/conv3d_3/BiasAdd/ReadVariableOp'^model_1/conv3d_3/Conv3D/ReadVariableOp(^model_1/conv3d_4/BiasAdd/ReadVariableOp'^model_1/conv3d_4/Conv3D/ReadVariableOp(^model_1/dense_12/BiasAdd/ReadVariableOp'^model_1/dense_12/MatMul/ReadVariableOp(^model_1/dense_13/BiasAdd/ReadVariableOp'^model_1/dense_13/MatMul/ReadVariableOp(^model_1/dense_14/BiasAdd/ReadVariableOp'^model_1/dense_14/MatMul/ReadVariableOp(^model_1/dense_15/BiasAdd/ReadVariableOp'^model_1/dense_15/MatMul/ReadVariableOp(^model_1/dense_16/BiasAdd/ReadVariableOp'^model_1/dense_16/MatMul/ReadVariableOp(^model_1/dense_17/BiasAdd/ReadVariableOp'^model_1/dense_17/MatMul/ReadVariableOp(^model_1/dense_18/BiasAdd/ReadVariableOp'^model_1/dense_18/MatMul/ReadVariableOp(^model_1/dense_19/BiasAdd/ReadVariableOp'^model_1/dense_19/MatMul/ReadVariableOp(^model_1/dense_20/BiasAdd/ReadVariableOp'^model_1/dense_20/MatMul/ReadVariableOp(^model_1/dense_21/BiasAdd/ReadVariableOp'^model_1/dense_21/MatMul/ReadVariableOp(^model_1/dense_22/BiasAdd/ReadVariableOp'^model_1/dense_22/MatMul/ReadVariableOp(^model_1/dense_23/BiasAdd/ReadVariableOp'^model_1/dense_23/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity_3?


Identity_4Identity#model_1/flatten_18/Reshape:output:0(^model_1/conv3d_3/BiasAdd/ReadVariableOp'^model_1/conv3d_3/Conv3D/ReadVariableOp(^model_1/conv3d_4/BiasAdd/ReadVariableOp'^model_1/conv3d_4/Conv3D/ReadVariableOp(^model_1/dense_12/BiasAdd/ReadVariableOp'^model_1/dense_12/MatMul/ReadVariableOp(^model_1/dense_13/BiasAdd/ReadVariableOp'^model_1/dense_13/MatMul/ReadVariableOp(^model_1/dense_14/BiasAdd/ReadVariableOp'^model_1/dense_14/MatMul/ReadVariableOp(^model_1/dense_15/BiasAdd/ReadVariableOp'^model_1/dense_15/MatMul/ReadVariableOp(^model_1/dense_16/BiasAdd/ReadVariableOp'^model_1/dense_16/MatMul/ReadVariableOp(^model_1/dense_17/BiasAdd/ReadVariableOp'^model_1/dense_17/MatMul/ReadVariableOp(^model_1/dense_18/BiasAdd/ReadVariableOp'^model_1/dense_18/MatMul/ReadVariableOp(^model_1/dense_19/BiasAdd/ReadVariableOp'^model_1/dense_19/MatMul/ReadVariableOp(^model_1/dense_20/BiasAdd/ReadVariableOp'^model_1/dense_20/MatMul/ReadVariableOp(^model_1/dense_21/BiasAdd/ReadVariableOp'^model_1/dense_21/MatMul/ReadVariableOp(^model_1/dense_22/BiasAdd/ReadVariableOp'^model_1/dense_22/MatMul/ReadVariableOp(^model_1/dense_23/BiasAdd/ReadVariableOp'^model_1/dense_23/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity_4?


Identity_5Identity#model_1/flatten_19/Reshape:output:0(^model_1/conv3d_3/BiasAdd/ReadVariableOp'^model_1/conv3d_3/Conv3D/ReadVariableOp(^model_1/conv3d_4/BiasAdd/ReadVariableOp'^model_1/conv3d_4/Conv3D/ReadVariableOp(^model_1/dense_12/BiasAdd/ReadVariableOp'^model_1/dense_12/MatMul/ReadVariableOp(^model_1/dense_13/BiasAdd/ReadVariableOp'^model_1/dense_13/MatMul/ReadVariableOp(^model_1/dense_14/BiasAdd/ReadVariableOp'^model_1/dense_14/MatMul/ReadVariableOp(^model_1/dense_15/BiasAdd/ReadVariableOp'^model_1/dense_15/MatMul/ReadVariableOp(^model_1/dense_16/BiasAdd/ReadVariableOp'^model_1/dense_16/MatMul/ReadVariableOp(^model_1/dense_17/BiasAdd/ReadVariableOp'^model_1/dense_17/MatMul/ReadVariableOp(^model_1/dense_18/BiasAdd/ReadVariableOp'^model_1/dense_18/MatMul/ReadVariableOp(^model_1/dense_19/BiasAdd/ReadVariableOp'^model_1/dense_19/MatMul/ReadVariableOp(^model_1/dense_20/BiasAdd/ReadVariableOp'^model_1/dense_20/MatMul/ReadVariableOp(^model_1/dense_21/BiasAdd/ReadVariableOp'^model_1/dense_21/MatMul/ReadVariableOp(^model_1/dense_22/BiasAdd/ReadVariableOp'^model_1/dense_22/MatMul/ReadVariableOp(^model_1/dense_23/BiasAdd/ReadVariableOp'^model_1/dense_23/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity_5?


Identity_6Identity#model_1/flatten_20/Reshape:output:0(^model_1/conv3d_3/BiasAdd/ReadVariableOp'^model_1/conv3d_3/Conv3D/ReadVariableOp(^model_1/conv3d_4/BiasAdd/ReadVariableOp'^model_1/conv3d_4/Conv3D/ReadVariableOp(^model_1/dense_12/BiasAdd/ReadVariableOp'^model_1/dense_12/MatMul/ReadVariableOp(^model_1/dense_13/BiasAdd/ReadVariableOp'^model_1/dense_13/MatMul/ReadVariableOp(^model_1/dense_14/BiasAdd/ReadVariableOp'^model_1/dense_14/MatMul/ReadVariableOp(^model_1/dense_15/BiasAdd/ReadVariableOp'^model_1/dense_15/MatMul/ReadVariableOp(^model_1/dense_16/BiasAdd/ReadVariableOp'^model_1/dense_16/MatMul/ReadVariableOp(^model_1/dense_17/BiasAdd/ReadVariableOp'^model_1/dense_17/MatMul/ReadVariableOp(^model_1/dense_18/BiasAdd/ReadVariableOp'^model_1/dense_18/MatMul/ReadVariableOp(^model_1/dense_19/BiasAdd/ReadVariableOp'^model_1/dense_19/MatMul/ReadVariableOp(^model_1/dense_20/BiasAdd/ReadVariableOp'^model_1/dense_20/MatMul/ReadVariableOp(^model_1/dense_21/BiasAdd/ReadVariableOp'^model_1/dense_21/MatMul/ReadVariableOp(^model_1/dense_22/BiasAdd/ReadVariableOp'^model_1/dense_22/MatMul/ReadVariableOp(^model_1/dense_23/BiasAdd/ReadVariableOp'^model_1/dense_23/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity_6?


Identity_7Identity#model_1/flatten_21/Reshape:output:0(^model_1/conv3d_3/BiasAdd/ReadVariableOp'^model_1/conv3d_3/Conv3D/ReadVariableOp(^model_1/conv3d_4/BiasAdd/ReadVariableOp'^model_1/conv3d_4/Conv3D/ReadVariableOp(^model_1/dense_12/BiasAdd/ReadVariableOp'^model_1/dense_12/MatMul/ReadVariableOp(^model_1/dense_13/BiasAdd/ReadVariableOp'^model_1/dense_13/MatMul/ReadVariableOp(^model_1/dense_14/BiasAdd/ReadVariableOp'^model_1/dense_14/MatMul/ReadVariableOp(^model_1/dense_15/BiasAdd/ReadVariableOp'^model_1/dense_15/MatMul/ReadVariableOp(^model_1/dense_16/BiasAdd/ReadVariableOp'^model_1/dense_16/MatMul/ReadVariableOp(^model_1/dense_17/BiasAdd/ReadVariableOp'^model_1/dense_17/MatMul/ReadVariableOp(^model_1/dense_18/BiasAdd/ReadVariableOp'^model_1/dense_18/MatMul/ReadVariableOp(^model_1/dense_19/BiasAdd/ReadVariableOp'^model_1/dense_19/MatMul/ReadVariableOp(^model_1/dense_20/BiasAdd/ReadVariableOp'^model_1/dense_20/MatMul/ReadVariableOp(^model_1/dense_21/BiasAdd/ReadVariableOp'^model_1/dense_21/MatMul/ReadVariableOp(^model_1/dense_22/BiasAdd/ReadVariableOp'^model_1/dense_22/MatMul/ReadVariableOp(^model_1/dense_23/BiasAdd/ReadVariableOp'^model_1/dense_23/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity_7?


Identity_8Identity#model_1/flatten_22/Reshape:output:0(^model_1/conv3d_3/BiasAdd/ReadVariableOp'^model_1/conv3d_3/Conv3D/ReadVariableOp(^model_1/conv3d_4/BiasAdd/ReadVariableOp'^model_1/conv3d_4/Conv3D/ReadVariableOp(^model_1/dense_12/BiasAdd/ReadVariableOp'^model_1/dense_12/MatMul/ReadVariableOp(^model_1/dense_13/BiasAdd/ReadVariableOp'^model_1/dense_13/MatMul/ReadVariableOp(^model_1/dense_14/BiasAdd/ReadVariableOp'^model_1/dense_14/MatMul/ReadVariableOp(^model_1/dense_15/BiasAdd/ReadVariableOp'^model_1/dense_15/MatMul/ReadVariableOp(^model_1/dense_16/BiasAdd/ReadVariableOp'^model_1/dense_16/MatMul/ReadVariableOp(^model_1/dense_17/BiasAdd/ReadVariableOp'^model_1/dense_17/MatMul/ReadVariableOp(^model_1/dense_18/BiasAdd/ReadVariableOp'^model_1/dense_18/MatMul/ReadVariableOp(^model_1/dense_19/BiasAdd/ReadVariableOp'^model_1/dense_19/MatMul/ReadVariableOp(^model_1/dense_20/BiasAdd/ReadVariableOp'^model_1/dense_20/MatMul/ReadVariableOp(^model_1/dense_21/BiasAdd/ReadVariableOp'^model_1/dense_21/MatMul/ReadVariableOp(^model_1/dense_22/BiasAdd/ReadVariableOp'^model_1/dense_22/MatMul/ReadVariableOp(^model_1/dense_23/BiasAdd/ReadVariableOp'^model_1/dense_23/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity_8?


Identity_9Identity#model_1/flatten_23/Reshape:output:0(^model_1/conv3d_3/BiasAdd/ReadVariableOp'^model_1/conv3d_3/Conv3D/ReadVariableOp(^model_1/conv3d_4/BiasAdd/ReadVariableOp'^model_1/conv3d_4/Conv3D/ReadVariableOp(^model_1/dense_12/BiasAdd/ReadVariableOp'^model_1/dense_12/MatMul/ReadVariableOp(^model_1/dense_13/BiasAdd/ReadVariableOp'^model_1/dense_13/MatMul/ReadVariableOp(^model_1/dense_14/BiasAdd/ReadVariableOp'^model_1/dense_14/MatMul/ReadVariableOp(^model_1/dense_15/BiasAdd/ReadVariableOp'^model_1/dense_15/MatMul/ReadVariableOp(^model_1/dense_16/BiasAdd/ReadVariableOp'^model_1/dense_16/MatMul/ReadVariableOp(^model_1/dense_17/BiasAdd/ReadVariableOp'^model_1/dense_17/MatMul/ReadVariableOp(^model_1/dense_18/BiasAdd/ReadVariableOp'^model_1/dense_18/MatMul/ReadVariableOp(^model_1/dense_19/BiasAdd/ReadVariableOp'^model_1/dense_19/MatMul/ReadVariableOp(^model_1/dense_20/BiasAdd/ReadVariableOp'^model_1/dense_20/MatMul/ReadVariableOp(^model_1/dense_21/BiasAdd/ReadVariableOp'^model_1/dense_21/MatMul/ReadVariableOp(^model_1/dense_22/BiasAdd/ReadVariableOp'^model_1/dense_22/MatMul/ReadVariableOp(^model_1/dense_23/BiasAdd/ReadVariableOp'^model_1/dense_23/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity_9?

Identity_10Identity#model_1/flatten_24/Reshape:output:0(^model_1/conv3d_3/BiasAdd/ReadVariableOp'^model_1/conv3d_3/Conv3D/ReadVariableOp(^model_1/conv3d_4/BiasAdd/ReadVariableOp'^model_1/conv3d_4/Conv3D/ReadVariableOp(^model_1/dense_12/BiasAdd/ReadVariableOp'^model_1/dense_12/MatMul/ReadVariableOp(^model_1/dense_13/BiasAdd/ReadVariableOp'^model_1/dense_13/MatMul/ReadVariableOp(^model_1/dense_14/BiasAdd/ReadVariableOp'^model_1/dense_14/MatMul/ReadVariableOp(^model_1/dense_15/BiasAdd/ReadVariableOp'^model_1/dense_15/MatMul/ReadVariableOp(^model_1/dense_16/BiasAdd/ReadVariableOp'^model_1/dense_16/MatMul/ReadVariableOp(^model_1/dense_17/BiasAdd/ReadVariableOp'^model_1/dense_17/MatMul/ReadVariableOp(^model_1/dense_18/BiasAdd/ReadVariableOp'^model_1/dense_18/MatMul/ReadVariableOp(^model_1/dense_19/BiasAdd/ReadVariableOp'^model_1/dense_19/MatMul/ReadVariableOp(^model_1/dense_20/BiasAdd/ReadVariableOp'^model_1/dense_20/MatMul/ReadVariableOp(^model_1/dense_21/BiasAdd/ReadVariableOp'^model_1/dense_21/MatMul/ReadVariableOp(^model_1/dense_22/BiasAdd/ReadVariableOp'^model_1/dense_22/MatMul/ReadVariableOp(^model_1/dense_23/BiasAdd/ReadVariableOp'^model_1/dense_23/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2
Identity_10?

Identity_11Identity#model_1/flatten_25/Reshape:output:0(^model_1/conv3d_3/BiasAdd/ReadVariableOp'^model_1/conv3d_3/Conv3D/ReadVariableOp(^model_1/conv3d_4/BiasAdd/ReadVariableOp'^model_1/conv3d_4/Conv3D/ReadVariableOp(^model_1/dense_12/BiasAdd/ReadVariableOp'^model_1/dense_12/MatMul/ReadVariableOp(^model_1/dense_13/BiasAdd/ReadVariableOp'^model_1/dense_13/MatMul/ReadVariableOp(^model_1/dense_14/BiasAdd/ReadVariableOp'^model_1/dense_14/MatMul/ReadVariableOp(^model_1/dense_15/BiasAdd/ReadVariableOp'^model_1/dense_15/MatMul/ReadVariableOp(^model_1/dense_16/BiasAdd/ReadVariableOp'^model_1/dense_16/MatMul/ReadVariableOp(^model_1/dense_17/BiasAdd/ReadVariableOp'^model_1/dense_17/MatMul/ReadVariableOp(^model_1/dense_18/BiasAdd/ReadVariableOp'^model_1/dense_18/MatMul/ReadVariableOp(^model_1/dense_19/BiasAdd/ReadVariableOp'^model_1/dense_19/MatMul/ReadVariableOp(^model_1/dense_20/BiasAdd/ReadVariableOp'^model_1/dense_20/MatMul/ReadVariableOp(^model_1/dense_21/BiasAdd/ReadVariableOp'^model_1/dense_21/MatMul/ReadVariableOp(^model_1/dense_22/BiasAdd/ReadVariableOp'^model_1/dense_22/MatMul/ReadVariableOp(^model_1/dense_23/BiasAdd/ReadVariableOp'^model_1/dense_23/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2
Identity_11"
identityIdentity:output:0"!

identity_1Identity_1:output:0"#
identity_10Identity_10:output:0"#
identity_11Identity_11:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"!

identity_6Identity_6:output:0"!

identity_7Identity_7:output:0"!

identity_8Identity_8:output:0"!

identity_9Identity_9:output:0*?
_input_shapes?
?:?????????ppd::::::::::::::::::::::::::::2R
'model_1/conv3d_3/BiasAdd/ReadVariableOp'model_1/conv3d_3/BiasAdd/ReadVariableOp2P
&model_1/conv3d_3/Conv3D/ReadVariableOp&model_1/conv3d_3/Conv3D/ReadVariableOp2R
'model_1/conv3d_4/BiasAdd/ReadVariableOp'model_1/conv3d_4/BiasAdd/ReadVariableOp2P
&model_1/conv3d_4/Conv3D/ReadVariableOp&model_1/conv3d_4/Conv3D/ReadVariableOp2R
'model_1/dense_12/BiasAdd/ReadVariableOp'model_1/dense_12/BiasAdd/ReadVariableOp2P
&model_1/dense_12/MatMul/ReadVariableOp&model_1/dense_12/MatMul/ReadVariableOp2R
'model_1/dense_13/BiasAdd/ReadVariableOp'model_1/dense_13/BiasAdd/ReadVariableOp2P
&model_1/dense_13/MatMul/ReadVariableOp&model_1/dense_13/MatMul/ReadVariableOp2R
'model_1/dense_14/BiasAdd/ReadVariableOp'model_1/dense_14/BiasAdd/ReadVariableOp2P
&model_1/dense_14/MatMul/ReadVariableOp&model_1/dense_14/MatMul/ReadVariableOp2R
'model_1/dense_15/BiasAdd/ReadVariableOp'model_1/dense_15/BiasAdd/ReadVariableOp2P
&model_1/dense_15/MatMul/ReadVariableOp&model_1/dense_15/MatMul/ReadVariableOp2R
'model_1/dense_16/BiasAdd/ReadVariableOp'model_1/dense_16/BiasAdd/ReadVariableOp2P
&model_1/dense_16/MatMul/ReadVariableOp&model_1/dense_16/MatMul/ReadVariableOp2R
'model_1/dense_17/BiasAdd/ReadVariableOp'model_1/dense_17/BiasAdd/ReadVariableOp2P
&model_1/dense_17/MatMul/ReadVariableOp&model_1/dense_17/MatMul/ReadVariableOp2R
'model_1/dense_18/BiasAdd/ReadVariableOp'model_1/dense_18/BiasAdd/ReadVariableOp2P
&model_1/dense_18/MatMul/ReadVariableOp&model_1/dense_18/MatMul/ReadVariableOp2R
'model_1/dense_19/BiasAdd/ReadVariableOp'model_1/dense_19/BiasAdd/ReadVariableOp2P
&model_1/dense_19/MatMul/ReadVariableOp&model_1/dense_19/MatMul/ReadVariableOp2R
'model_1/dense_20/BiasAdd/ReadVariableOp'model_1/dense_20/BiasAdd/ReadVariableOp2P
&model_1/dense_20/MatMul/ReadVariableOp&model_1/dense_20/MatMul/ReadVariableOp2R
'model_1/dense_21/BiasAdd/ReadVariableOp'model_1/dense_21/BiasAdd/ReadVariableOp2P
&model_1/dense_21/MatMul/ReadVariableOp&model_1/dense_21/MatMul/ReadVariableOp2R
'model_1/dense_22/BiasAdd/ReadVariableOp'model_1/dense_22/BiasAdd/ReadVariableOp2P
&model_1/dense_22/MatMul/ReadVariableOp&model_1/dense_22/MatMul/ReadVariableOp2R
'model_1/dense_23/BiasAdd/ReadVariableOp'model_1/dense_23/BiasAdd/ReadVariableOp2P
&model_1/dense_23/MatMul/ReadVariableOp&model_1/dense_23/MatMul/ReadVariableOp:\ X
3
_output_shapes!
:?????????ppd
!
_user_specified_name	input_2
?	
?
C__inference_dense_19_layer_call_and_return_conditional_losses_10066

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?	
?
B__inference_dense_23_layer_call_and_return_conditional_losses_8363

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:?????????2	
Sigmoid?
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
`
D__inference_flatten_14_layer_call_and_return_conditional_losses_8836

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
Constg
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:?????????2	
Reshaped
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
C__inference_dense_23_layer_call_and_return_conditional_losses_10146

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:?????????2	
Sigmoid?
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
F
*__inference_flatten_22_layer_call_fn_10254

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_flatten_22_layer_call_and_return_conditional_losses_87242
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
`
D__inference_flatten_16_layer_call_and_return_conditional_losses_8808

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
Constg
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:?????????2	
Reshaped
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
B__inference_dense_14_layer_call_and_return_conditional_losses_8606

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
`
D__inference_flatten_13_layer_call_and_return_conditional_losses_9910

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"????h  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:??????????2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*2
_input_shapes!
:?????????
:[ W
3
_output_shapes!
:?????????

 
_user_specified_nameinputs
?	
?
B__inference_dense_21_layer_call_and_return_conditional_losses_8417

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
|
'__inference_conv3d_4_layer_call_fn_9904

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_conv3d_4_layer_call_and_return_conditional_losses_83212
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*3
_output_shapes!
:?????????
2

Identity"
identityIdentity:output:0*:
_input_shapes)
':?????????::22
StatefulPartitionedCallStatefulPartitionedCall:[ W
3
_output_shapes!
:?????????
 
_user_specified_nameinputs
?	
?
C__inference_dense_17_layer_call_and_return_conditional_losses_10026

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
F
*__inference_flatten_18_layer_call_fn_10210

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_flatten_18_layer_call_and_return_conditional_losses_87802
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
B__inference_dense_15_layer_call_and_return_conditional_losses_9986

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
J
.__inference_max_pooling3d_3_layer_call_fn_8278

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *W
_output_shapesE
C:A?????????????????????????????????????????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_max_pooling3d_3_layer_call_and_return_conditional_losses_82722
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*W
_output_shapesE
C:A?????????????????????????????????????????????2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:A?????????????????????????????????????????????: {
W
_output_shapesE
C:A?????????????????????????????????????????????
 
_user_specified_nameinputs
?
F
*__inference_flatten_17_layer_call_fn_10199

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_flatten_17_layer_call_and_return_conditional_losses_87942
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
`
D__inference_flatten_19_layer_call_and_return_conditional_losses_8766

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
Constg
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:?????????2	
Reshaped
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
|
'__inference_dense_14_layer_call_fn_9975

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_14_layer_call_and_return_conditional_losses_86062
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
? 
?
&__inference_model_1_layer_call_fn_9323
input_2
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26
identity

identity_1

identity_2

identity_3

identity_4

identity_5

identity_6

identity_7

identity_8

identity_9
identity_10
identity_11??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26*(
Tin!
2*
Tout
2*
_collective_manager_ids
 *?
_output_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????*>
_read_only_resource_inputs 
	
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_model_1_layer_call_and_return_conditional_losses_92422
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity?

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_1?

Identity_2Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_2?

Identity_3Identity StatefulPartitionedCall:output:3^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_3?

Identity_4Identity StatefulPartitionedCall:output:4^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_4?

Identity_5Identity StatefulPartitionedCall:output:5^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_5?

Identity_6Identity StatefulPartitionedCall:output:6^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_6?

Identity_7Identity StatefulPartitionedCall:output:7^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_7?

Identity_8Identity StatefulPartitionedCall:output:8^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_8?

Identity_9Identity StatefulPartitionedCall:output:9^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_9?
Identity_10Identity!StatefulPartitionedCall:output:10^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2
Identity_10?
Identity_11Identity!StatefulPartitionedCall:output:11^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2
Identity_11"
identityIdentity:output:0"!

identity_1Identity_1:output:0"#
identity_10Identity_10:output:0"#
identity_11Identity_11:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"!

identity_6Identity_6:output:0"!

identity_7Identity_7:output:0"!

identity_8Identity_8:output:0"!

identity_9Identity_9:output:0*?
_input_shapes?
?:?????????ppd::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:\ X
3
_output_shapes!
:?????????ppd
!
_user_specified_name	input_2
?	
?
C__inference_dense_16_layer_call_and_return_conditional_losses_10006

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
a
E__inference_flatten_21_layer_call_and_return_conditional_losses_10238

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
Constg
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:?????????2	
Reshaped
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
a
E__inference_flatten_17_layer_call_and_return_conditional_losses_10194

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
Constg
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:?????????2	
Reshaped
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
B__inference_dense_12_layer_call_and_return_conditional_losses_9926

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
}
(__inference_dense_20_layer_call_fn_10095

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_20_layer_call_and_return_conditional_losses_84442
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?

?
B__inference_conv3d_3_layer_call_and_return_conditional_losses_9875

inputs"
conv3d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv3D/ReadVariableOp?
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:d*
dtype02
Conv3D/ReadVariableOp?
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????66*
paddingVALID*
strides	
2
Conv3D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:?????????662	
BiasAddd
ReluReluBiasAdd:output:0*
T0*3
_output_shapes!
:?????????662
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*
T0*3
_output_shapes!
:?????????662

Identity"
identityIdentity:output:0*:
_input_shapes)
':?????????ppd::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:[ W
3
_output_shapes!
:?????????ppd
 
_user_specified_nameinputs
?	
?
B__inference_dense_16_layer_call_and_return_conditional_losses_8552

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
}
(__inference_dense_17_layer_call_fn_10035

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_17_layer_call_and_return_conditional_losses_85252
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
`
D__inference_flatten_17_layer_call_and_return_conditional_losses_8794

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
Constg
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:?????????2	
Reshaped
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
G
input_2<
serving_default_input_2:0?????????ppd>

flatten_140
StatefulPartitionedCall:0?????????>

flatten_150
StatefulPartitionedCall:1?????????>

flatten_160
StatefulPartitionedCall:2?????????>

flatten_170
StatefulPartitionedCall:3?????????>

flatten_180
StatefulPartitionedCall:4?????????>

flatten_190
StatefulPartitionedCall:5?????????>

flatten_200
StatefulPartitionedCall:6?????????>

flatten_210
StatefulPartitionedCall:7?????????>

flatten_220
StatefulPartitionedCall:8?????????>

flatten_230
StatefulPartitionedCall:9??????????

flatten_241
StatefulPartitionedCall:10??????????

flatten_251
StatefulPartitionedCall:11?????????tensorflow/serving/predict:̿
??
layer-0
layer_with_weights-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer-4
layer-5
layer_with_weights-2
layer-6
layer_with_weights-3
layer-7
	layer_with_weights-4
	layer-8

layer_with_weights-5

layer-9
layer_with_weights-6
layer-10
layer_with_weights-7
layer-11
layer_with_weights-8
layer-12
layer_with_weights-9
layer-13
layer_with_weights-10
layer-14
layer_with_weights-11
layer-15
layer_with_weights-12
layer-16
layer_with_weights-13
layer-17
layer-18
layer-19
layer-20
layer-21
layer-22
layer-23
layer-24
layer-25
layer-26
layer-27
layer-28
layer-29
	optimizer
 loss
!trainable_variables
"	variables
#regularization_losses
$	keras_api
%
signatures
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses"??
_tf_keras_network??{"class_name": "Functional", "name": "model_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "model_1", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 25, 112, 112, 100]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_2"}, "name": "input_2", "inbound_nodes": []}, {"class_name": "Conv3D", "config": {"name": "conv3d_3", "trainable": true, "dtype": "float32", "filters": 20, "kernel_size": {"class_name": "__tuple__", "items": [5, 5, 5]}, "strides": {"class_name": "__tuple__", "items": [2, 2, 2]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv3d_3", "inbound_nodes": [[["input_2", 0, 0, {}]]]}, {"class_name": "MaxPooling3D", "config": {"name": "max_pooling3d_2", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2, 2]}, "data_format": "channels_last"}, "name": "max_pooling3d_2", "inbound_nodes": [[["conv3d_3", 0, 0, {}]]]}, {"class_name": "Conv3D", "config": {"name": "conv3d_4", "trainable": true, "dtype": "float32", "filters": 10, "kernel_size": {"class_name": "__tuple__", "items": [4, 4, 4]}, "strides": {"class_name": "__tuple__", "items": [2, 2, 2]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv3d_4", "inbound_nodes": [[["max_pooling3d_2", 0, 0, {}]]]}, {"class_name": "MaxPooling3D", "config": {"name": "max_pooling3d_3", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [1, 2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [1, 2, 2]}, "data_format": "channels_last"}, "name": "max_pooling3d_3", "inbound_nodes": [[["conv3d_4", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten_13", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_13", "inbound_nodes": [[["max_pooling3d_3", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_12", "trainable": true, "dtype": "float32", "units": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_12", "inbound_nodes": [[["flatten_13", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_13", "trainable": true, "dtype": "float32", "units": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_13", "inbound_nodes": [[["flatten_13", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_14", "trainable": true, "dtype": "float32", "units": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_14", "inbound_nodes": [[["flatten_13", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_15", "trainable": true, "dtype": "float32", "units": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_15", "inbound_nodes": [[["flatten_13", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_16", "trainable": true, "dtype": "float32", "units": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_16", "inbound_nodes": [[["flatten_13", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_17", "trainable": true, "dtype": "float32", "units": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_17", "inbound_nodes": [[["flatten_13", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_18", "trainable": true, "dtype": "float32", "units": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_18", "inbound_nodes": [[["flatten_13", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_19", "trainable": true, "dtype": "float32", "units": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_19", "inbound_nodes": [[["flatten_13", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_20", "trainable": true, "dtype": "float32", "units": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_20", "inbound_nodes": [[["flatten_13", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_21", "trainable": true, "dtype": "float32", "units": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_21", "inbound_nodes": [[["flatten_13", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_22", "trainable": true, "dtype": "float32", "units": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_22", "inbound_nodes": [[["flatten_13", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_23", "trainable": true, "dtype": "float32", "units": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_23", "inbound_nodes": [[["flatten_13", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten_14", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_14", "inbound_nodes": [[["dense_12", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten_15", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_15", "inbound_nodes": [[["dense_13", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten_16", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_16", "inbound_nodes": [[["dense_14", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten_17", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_17", "inbound_nodes": [[["dense_15", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten_18", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_18", "inbound_nodes": [[["dense_16", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten_19", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_19", "inbound_nodes": [[["dense_17", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten_20", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_20", "inbound_nodes": [[["dense_18", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten_21", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_21", "inbound_nodes": [[["dense_19", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten_22", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_22", "inbound_nodes": [[["dense_20", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten_23", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_23", "inbound_nodes": [[["dense_21", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten_24", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_24", "inbound_nodes": [[["dense_22", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten_25", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_25", "inbound_nodes": [[["dense_23", 0, 0, {}]]]}], "input_layers": [["input_2", 0, 0]], "output_layers": [["flatten_14", 0, 0], ["flatten_15", 0, 0], ["flatten_16", 0, 0], ["flatten_17", 0, 0], ["flatten_18", 0, 0], ["flatten_19", 0, 0], ["flatten_20", 0, 0], ["flatten_21", 0, 0], ["flatten_22", 0, 0], ["flatten_23", 0, 0], ["flatten_24", 0, 0], ["flatten_25", 0, 0]]}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 25, 112, 112, 100]}, "ndim": 5, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 25, 112, 112, 100]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "model_1", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 25, 112, 112, 100]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_2"}, "name": "input_2", "inbound_nodes": []}, {"class_name": "Conv3D", "config": {"name": "conv3d_3", "trainable": true, "dtype": "float32", "filters": 20, "kernel_size": {"class_name": "__tuple__", "items": [5, 5, 5]}, "strides": {"class_name": "__tuple__", "items": [2, 2, 2]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv3d_3", "inbound_nodes": [[["input_2", 0, 0, {}]]]}, {"class_name": "MaxPooling3D", "config": {"name": "max_pooling3d_2", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2, 2]}, "data_format": "channels_last"}, "name": "max_pooling3d_2", "inbound_nodes": [[["conv3d_3", 0, 0, {}]]]}, {"class_name": "Conv3D", "config": {"name": "conv3d_4", "trainable": true, "dtype": "float32", "filters": 10, "kernel_size": {"class_name": "__tuple__", "items": [4, 4, 4]}, "strides": {"class_name": "__tuple__", "items": [2, 2, 2]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv3d_4", "inbound_nodes": [[["max_pooling3d_2", 0, 0, {}]]]}, {"class_name": "MaxPooling3D", "config": {"name": "max_pooling3d_3", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [1, 2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [1, 2, 2]}, "data_format": "channels_last"}, "name": "max_pooling3d_3", "inbound_nodes": [[["conv3d_4", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten_13", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_13", "inbound_nodes": [[["max_pooling3d_3", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_12", "trainable": true, "dtype": "float32", "units": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_12", "inbound_nodes": [[["flatten_13", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_13", "trainable": true, "dtype": "float32", "units": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_13", "inbound_nodes": [[["flatten_13", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_14", "trainable": true, "dtype": "float32", "units": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_14", "inbound_nodes": [[["flatten_13", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_15", "trainable": true, "dtype": "float32", "units": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_15", "inbound_nodes": [[["flatten_13", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_16", "trainable": true, "dtype": "float32", "units": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_16", "inbound_nodes": [[["flatten_13", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_17", "trainable": true, "dtype": "float32", "units": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_17", "inbound_nodes": [[["flatten_13", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_18", "trainable": true, "dtype": "float32", "units": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_18", "inbound_nodes": [[["flatten_13", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_19", "trainable": true, "dtype": "float32", "units": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_19", "inbound_nodes": [[["flatten_13", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_20", "trainable": true, "dtype": "float32", "units": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_20", "inbound_nodes": [[["flatten_13", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_21", "trainable": true, "dtype": "float32", "units": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_21", "inbound_nodes": [[["flatten_13", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_22", "trainable": true, "dtype": "float32", "units": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_22", "inbound_nodes": [[["flatten_13", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_23", "trainable": true, "dtype": "float32", "units": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_23", "inbound_nodes": [[["flatten_13", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten_14", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_14", "inbound_nodes": [[["dense_12", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten_15", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_15", "inbound_nodes": [[["dense_13", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten_16", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_16", "inbound_nodes": [[["dense_14", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten_17", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_17", "inbound_nodes": [[["dense_15", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten_18", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_18", "inbound_nodes": [[["dense_16", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten_19", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_19", "inbound_nodes": [[["dense_17", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten_20", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_20", "inbound_nodes": [[["dense_18", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten_21", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_21", "inbound_nodes": [[["dense_19", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten_22", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_22", "inbound_nodes": [[["dense_20", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten_23", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_23", "inbound_nodes": [[["dense_21", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten_24", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_24", "inbound_nodes": [[["dense_22", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten_25", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_25", "inbound_nodes": [[["dense_23", 0, 0, {}]]]}], "input_layers": [["input_2", 0, 0]], "output_layers": [["flatten_14", 0, 0], ["flatten_15", 0, 0], ["flatten_16", 0, 0], ["flatten_17", 0, 0], ["flatten_18", 0, 0], ["flatten_19", 0, 0], ["flatten_20", 0, 0], ["flatten_21", 0, 0], ["flatten_22", 0, 0], ["flatten_23", 0, 0], ["flatten_24", 0, 0], ["flatten_25", 0, 0]]}}, "training_config": {"loss": ["mean_squared_error", "mean_squared_error", "mean_squared_error", "mean_squared_error", "mean_squared_error", "mean_squared_error", "mean_squared_error", "mean_squared_error", "mean_squared_error", "mean_squared_error", "mean_squared_error", "binary_crossentropy"], "metrics": null, "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 9.999999974752427e-07, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
?"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "input_2", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 25, 112, 112, 100]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 25, 112, 112, 100]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_2"}}
?


&kernel
'bias
(trainable_variables
)	variables
*regularization_losses
+	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Conv3D", "name": "conv3d_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv3d_3", "trainable": true, "dtype": "float32", "filters": 20, "kernel_size": {"class_name": "__tuple__", "items": [5, 5, 5]}, "strides": {"class_name": "__tuple__", "items": [2, 2, 2]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 5, "axes": {"-1": 100}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 25, 112, 112, 100]}}
?
,trainable_variables
-	variables
.regularization_losses
/	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "MaxPooling3D", "name": "max_pooling3d_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "max_pooling3d_2", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 5, "max_ndim": null, "min_ndim": null, "axes": {}}}}
?


0kernel
1bias
2trainable_variables
3	variables
4regularization_losses
5	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Conv3D", "name": "conv3d_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv3d_4", "trainable": true, "dtype": "float32", "filters": 10, "kernel_size": {"class_name": "__tuple__", "items": [4, 4, 4]}, "strides": {"class_name": "__tuple__", "items": [2, 2, 2]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 5, "axes": {"-1": 20}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 5, 27, 27, 20]}}
?
6trainable_variables
7	variables
8regularization_losses
9	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "MaxPooling3D", "name": "max_pooling3d_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "max_pooling3d_3", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [1, 2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [1, 2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 5, "max_ndim": null, "min_ndim": null, "axes": {}}}}
?
:trainable_variables
;	variables
<regularization_losses
=	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Flatten", "name": "flatten_13", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "flatten_13", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
?

>kernel
?bias
@trainable_variables
A	variables
Bregularization_losses
C	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_12", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_12", "trainable": true, "dtype": "float32", "units": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 360}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 360]}}
?

Dkernel
Ebias
Ftrainable_variables
G	variables
Hregularization_losses
I	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_13", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_13", "trainable": true, "dtype": "float32", "units": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 360}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 360]}}
?

Jkernel
Kbias
Ltrainable_variables
M	variables
Nregularization_losses
O	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_14", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_14", "trainable": true, "dtype": "float32", "units": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 360}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 360]}}
?

Pkernel
Qbias
Rtrainable_variables
S	variables
Tregularization_losses
U	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_15", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_15", "trainable": true, "dtype": "float32", "units": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 360}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 360]}}
?

Vkernel
Wbias
Xtrainable_variables
Y	variables
Zregularization_losses
[	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_16", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_16", "trainable": true, "dtype": "float32", "units": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 360}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 360]}}
?

\kernel
]bias
^trainable_variables
_	variables
`regularization_losses
a	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_17", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_17", "trainable": true, "dtype": "float32", "units": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 360}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 360]}}
?

bkernel
cbias
dtrainable_variables
e	variables
fregularization_losses
g	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_18", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_18", "trainable": true, "dtype": "float32", "units": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 360}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 360]}}
?

hkernel
ibias
jtrainable_variables
k	variables
lregularization_losses
m	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_19", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_19", "trainable": true, "dtype": "float32", "units": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 360}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 360]}}
?

nkernel
obias
ptrainable_variables
q	variables
rregularization_losses
s	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_20", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_20", "trainable": true, "dtype": "float32", "units": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 360}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 360]}}
?

tkernel
ubias
vtrainable_variables
w	variables
xregularization_losses
y	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_21", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_21", "trainable": true, "dtype": "float32", "units": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 360}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 360]}}
?

zkernel
{bias
|trainable_variables
}	variables
~regularization_losses
	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_22", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_22", "trainable": true, "dtype": "float32", "units": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 360}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 360]}}
?
?kernel
	?bias
?trainable_variables
?	variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_23", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_23", "trainable": true, "dtype": "float32", "units": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 360}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 360]}}
?
?trainable_variables
?	variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Flatten", "name": "flatten_14", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "flatten_14", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
?
?trainable_variables
?	variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Flatten", "name": "flatten_15", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "flatten_15", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
?
?trainable_variables
?	variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Flatten", "name": "flatten_16", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "flatten_16", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
?
?trainable_variables
?	variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Flatten", "name": "flatten_17", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "flatten_17", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
?
?trainable_variables
?	variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Flatten", "name": "flatten_18", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "flatten_18", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
?
?trainable_variables
?	variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Flatten", "name": "flatten_19", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "flatten_19", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
?
?trainable_variables
?	variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Flatten", "name": "flatten_20", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "flatten_20", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
?
?trainable_variables
?	variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Flatten", "name": "flatten_21", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "flatten_21", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
?
?trainable_variables
?	variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Flatten", "name": "flatten_22", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "flatten_22", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
?
?trainable_variables
?	variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Flatten", "name": "flatten_23", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "flatten_23", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
?
?trainable_variables
?	variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Flatten", "name": "flatten_24", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "flatten_24", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
?
?trainable_variables
?	variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Flatten", "name": "flatten_25", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "flatten_25", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
?
	?iter
?beta_1
?beta_2

?decay
?learning_rate&m?'m?0m?1m?>m??m?Dm?Em?Jm?Km?Pm?Qm?Vm?Wm?\m?]m?bm?cm?hm?im?nm?om?tm?um?zm?{m?	?m?	?m?&v?'v?0v?1v?>v??v?Dv?Ev?Jv?Kv?Pv?Qv?Vv?Wv?\v?]v?bv?cv?hv?iv?nv?ov?tv?uv?zv?{v?	?v?	?v?"
	optimizer
 "
trackable_list_wrapper
?
&0
'1
02
13
>4
?5
D6
E7
J8
K9
P10
Q11
V12
W13
\14
]15
b16
c17
h18
i19
n20
o21
t22
u23
z24
{25
?26
?27"
trackable_list_wrapper
?
&0
'1
02
13
>4
?5
D6
E7
J8
K9
P10
Q11
V12
W13
\14
]15
b16
c17
h18
i19
n20
o21
t22
u23
z24
{25
?26
?27"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layer_metrics
!trainable_variables
"	variables
?non_trainable_variables
?metrics
#regularization_losses
 ?layer_regularization_losses
?layers
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
-
?serving_default"
signature_map
-:+d2conv3d_3/kernel
:2conv3d_3/bias
.
&0
'1"
trackable_list_wrapper
.
&0
'1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layer_metrics
(trainable_variables
)	variables
?non_trainable_variables
?metrics
*regularization_losses
 ?layer_regularization_losses
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layer_metrics
,trainable_variables
-	variables
?non_trainable_variables
?metrics
.regularization_losses
 ?layer_regularization_losses
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
-:+
2conv3d_4/kernel
:
2conv3d_4/bias
.
00
11"
trackable_list_wrapper
.
00
11"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layer_metrics
2trainable_variables
3	variables
?non_trainable_variables
?metrics
4regularization_losses
 ?layer_regularization_losses
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layer_metrics
6trainable_variables
7	variables
?non_trainable_variables
?metrics
8regularization_losses
 ?layer_regularization_losses
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layer_metrics
:trainable_variables
;	variables
?non_trainable_variables
?metrics
<regularization_losses
 ?layer_regularization_losses
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
": 	?2dense_12/kernel
:2dense_12/bias
.
>0
?1"
trackable_list_wrapper
.
>0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layer_metrics
@trainable_variables
A	variables
?non_trainable_variables
?metrics
Bregularization_losses
 ?layer_regularization_losses
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
": 	?2dense_13/kernel
:2dense_13/bias
.
D0
E1"
trackable_list_wrapper
.
D0
E1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layer_metrics
Ftrainable_variables
G	variables
?non_trainable_variables
?metrics
Hregularization_losses
 ?layer_regularization_losses
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
": 	?2dense_14/kernel
:2dense_14/bias
.
J0
K1"
trackable_list_wrapper
.
J0
K1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layer_metrics
Ltrainable_variables
M	variables
?non_trainable_variables
?metrics
Nregularization_losses
 ?layer_regularization_losses
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
": 	?2dense_15/kernel
:2dense_15/bias
.
P0
Q1"
trackable_list_wrapper
.
P0
Q1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layer_metrics
Rtrainable_variables
S	variables
?non_trainable_variables
?metrics
Tregularization_losses
 ?layer_regularization_losses
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
": 	?2dense_16/kernel
:2dense_16/bias
.
V0
W1"
trackable_list_wrapper
.
V0
W1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layer_metrics
Xtrainable_variables
Y	variables
?non_trainable_variables
?metrics
Zregularization_losses
 ?layer_regularization_losses
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
": 	?2dense_17/kernel
:2dense_17/bias
.
\0
]1"
trackable_list_wrapper
.
\0
]1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layer_metrics
^trainable_variables
_	variables
?non_trainable_variables
?metrics
`regularization_losses
 ?layer_regularization_losses
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
": 	?2dense_18/kernel
:2dense_18/bias
.
b0
c1"
trackable_list_wrapper
.
b0
c1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layer_metrics
dtrainable_variables
e	variables
?non_trainable_variables
?metrics
fregularization_losses
 ?layer_regularization_losses
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
": 	?2dense_19/kernel
:2dense_19/bias
.
h0
i1"
trackable_list_wrapper
.
h0
i1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layer_metrics
jtrainable_variables
k	variables
?non_trainable_variables
?metrics
lregularization_losses
 ?layer_regularization_losses
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
": 	?2dense_20/kernel
:2dense_20/bias
.
n0
o1"
trackable_list_wrapper
.
n0
o1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layer_metrics
ptrainable_variables
q	variables
?non_trainable_variables
?metrics
rregularization_losses
 ?layer_regularization_losses
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
": 	?2dense_21/kernel
:2dense_21/bias
.
t0
u1"
trackable_list_wrapper
.
t0
u1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layer_metrics
vtrainable_variables
w	variables
?non_trainable_variables
?metrics
xregularization_losses
 ?layer_regularization_losses
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
": 	?2dense_22/kernel
:2dense_22/bias
.
z0
{1"
trackable_list_wrapper
.
z0
{1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layer_metrics
|trainable_variables
}	variables
?non_trainable_variables
?metrics
~regularization_losses
 ?layer_regularization_losses
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
": 	?2dense_23/kernel
:2dense_23/bias
0
?0
?1"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layer_metrics
?trainable_variables
?	variables
?non_trainable_variables
?metrics
?regularization_losses
 ?layer_regularization_losses
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layer_metrics
?trainable_variables
?	variables
?non_trainable_variables
?metrics
?regularization_losses
 ?layer_regularization_losses
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layer_metrics
?trainable_variables
?	variables
?non_trainable_variables
?metrics
?regularization_losses
 ?layer_regularization_losses
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layer_metrics
?trainable_variables
?	variables
?non_trainable_variables
?metrics
?regularization_losses
 ?layer_regularization_losses
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layer_metrics
?trainable_variables
?	variables
?non_trainable_variables
?metrics
?regularization_losses
 ?layer_regularization_losses
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layer_metrics
?trainable_variables
?	variables
?non_trainable_variables
?metrics
?regularization_losses
 ?layer_regularization_losses
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layer_metrics
?trainable_variables
?	variables
?non_trainable_variables
?metrics
?regularization_losses
 ?layer_regularization_losses
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layer_metrics
?trainable_variables
?	variables
?non_trainable_variables
?metrics
?regularization_losses
 ?layer_regularization_losses
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layer_metrics
?trainable_variables
?	variables
?non_trainable_variables
?metrics
?regularization_losses
 ?layer_regularization_losses
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layer_metrics
?trainable_variables
?	variables
?non_trainable_variables
?metrics
?regularization_losses
 ?layer_regularization_losses
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layer_metrics
?trainable_variables
?	variables
?non_trainable_variables
?metrics
?regularization_losses
 ?layer_regularization_losses
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layer_metrics
?trainable_variables
?	variables
?non_trainable_variables
?metrics
?regularization_losses
 ?layer_regularization_losses
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layer_metrics
?trainable_variables
?	variables
?non_trainable_variables
?metrics
?regularization_losses
 ?layer_regularization_losses
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
?
?0
?1
?2
?3
?4
?5
?6
?7
?8
?9
?10
?11
?12"
trackable_list_wrapper
 "
trackable_list_wrapper
?
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
29"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?

?total

?count
?	variables
?	keras_api"?
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}
?

?total

?count
?	variables
?	keras_api"?
_tf_keras_metric?{"class_name": "Mean", "name": "flatten_14_loss", "dtype": "float32", "config": {"name": "flatten_14_loss", "dtype": "float32"}}
?

?total

?count
?	variables
?	keras_api"?
_tf_keras_metric?{"class_name": "Mean", "name": "flatten_15_loss", "dtype": "float32", "config": {"name": "flatten_15_loss", "dtype": "float32"}}
?

?total

?count
?	variables
?	keras_api"?
_tf_keras_metric?{"class_name": "Mean", "name": "flatten_16_loss", "dtype": "float32", "config": {"name": "flatten_16_loss", "dtype": "float32"}}
?

?total

?count
?	variables
?	keras_api"?
_tf_keras_metric?{"class_name": "Mean", "name": "flatten_17_loss", "dtype": "float32", "config": {"name": "flatten_17_loss", "dtype": "float32"}}
?

?total

?count
?	variables
?	keras_api"?
_tf_keras_metric?{"class_name": "Mean", "name": "flatten_18_loss", "dtype": "float32", "config": {"name": "flatten_18_loss", "dtype": "float32"}}
?

?total

?count
?	variables
?	keras_api"?
_tf_keras_metric?{"class_name": "Mean", "name": "flatten_19_loss", "dtype": "float32", "config": {"name": "flatten_19_loss", "dtype": "float32"}}
?

?total

?count
?	variables
?	keras_api"?
_tf_keras_metric?{"class_name": "Mean", "name": "flatten_20_loss", "dtype": "float32", "config": {"name": "flatten_20_loss", "dtype": "float32"}}
?

?total

?count
?	variables
?	keras_api"?
_tf_keras_metric?{"class_name": "Mean", "name": "flatten_21_loss", "dtype": "float32", "config": {"name": "flatten_21_loss", "dtype": "float32"}}
?

?total

?count
?	variables
?	keras_api"?
_tf_keras_metric?{"class_name": "Mean", "name": "flatten_22_loss", "dtype": "float32", "config": {"name": "flatten_22_loss", "dtype": "float32"}}
?

?total

?count
?	variables
?	keras_api"?
_tf_keras_metric?{"class_name": "Mean", "name": "flatten_23_loss", "dtype": "float32", "config": {"name": "flatten_23_loss", "dtype": "float32"}}
?

?total

?count
?	variables
?	keras_api"?
_tf_keras_metric?{"class_name": "Mean", "name": "flatten_24_loss", "dtype": "float32", "config": {"name": "flatten_24_loss", "dtype": "float32"}}
?

?total

?count
?	variables
?	keras_api"?
_tf_keras_metric?{"class_name": "Mean", "name": "flatten_25_loss", "dtype": "float32", "config": {"name": "flatten_25_loss", "dtype": "float32"}}
:  (2total
:  (2count
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
:  (2total
:  (2count
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
:  (2total
:  (2count
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
:  (2total
:  (2count
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
:  (2total
:  (2count
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
:  (2total
:  (2count
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
:  (2total
:  (2count
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
:  (2total
:  (2count
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
:  (2total
:  (2count
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
:  (2total
:  (2count
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
:  (2total
:  (2count
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
:  (2total
:  (2count
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
:  (2total
:  (2count
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
2:0d2Adam/conv3d_3/kernel/m
 :2Adam/conv3d_3/bias/m
2:0
2Adam/conv3d_4/kernel/m
 :
2Adam/conv3d_4/bias/m
':%	?2Adam/dense_12/kernel/m
 :2Adam/dense_12/bias/m
':%	?2Adam/dense_13/kernel/m
 :2Adam/dense_13/bias/m
':%	?2Adam/dense_14/kernel/m
 :2Adam/dense_14/bias/m
':%	?2Adam/dense_15/kernel/m
 :2Adam/dense_15/bias/m
':%	?2Adam/dense_16/kernel/m
 :2Adam/dense_16/bias/m
':%	?2Adam/dense_17/kernel/m
 :2Adam/dense_17/bias/m
':%	?2Adam/dense_18/kernel/m
 :2Adam/dense_18/bias/m
':%	?2Adam/dense_19/kernel/m
 :2Adam/dense_19/bias/m
':%	?2Adam/dense_20/kernel/m
 :2Adam/dense_20/bias/m
':%	?2Adam/dense_21/kernel/m
 :2Adam/dense_21/bias/m
':%	?2Adam/dense_22/kernel/m
 :2Adam/dense_22/bias/m
':%	?2Adam/dense_23/kernel/m
 :2Adam/dense_23/bias/m
2:0d2Adam/conv3d_3/kernel/v
 :2Adam/conv3d_3/bias/v
2:0
2Adam/conv3d_4/kernel/v
 :
2Adam/conv3d_4/bias/v
':%	?2Adam/dense_12/kernel/v
 :2Adam/dense_12/bias/v
':%	?2Adam/dense_13/kernel/v
 :2Adam/dense_13/bias/v
':%	?2Adam/dense_14/kernel/v
 :2Adam/dense_14/bias/v
':%	?2Adam/dense_15/kernel/v
 :2Adam/dense_15/bias/v
':%	?2Adam/dense_16/kernel/v
 :2Adam/dense_16/bias/v
':%	?2Adam/dense_17/kernel/v
 :2Adam/dense_17/bias/v
':%	?2Adam/dense_18/kernel/v
 :2Adam/dense_18/bias/v
':%	?2Adam/dense_19/kernel/v
 :2Adam/dense_19/bias/v
':%	?2Adam/dense_20/kernel/v
 :2Adam/dense_20/bias/v
':%	?2Adam/dense_21/kernel/v
 :2Adam/dense_21/bias/v
':%	?2Adam/dense_22/kernel/v
 :2Adam/dense_22/bias/v
':%	?2Adam/dense_23/kernel/v
 :2Adam/dense_23/bias/v
?2?
&__inference_model_1_layer_call_fn_9323
&__inference_model_1_layer_call_fn_9864
&__inference_model_1_layer_call_fn_9781
&__inference_model_1_layer_call_fn_9140?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
__inference__wrapped_model_8254?
???
FullArgSpec
args? 
varargsjargs
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *2?/
-?*
input_2?????????ppd
?2?
A__inference_model_1_layer_call_and_return_conditional_losses_8956
A__inference_model_1_layer_call_and_return_conditional_losses_9698
A__inference_model_1_layer_call_and_return_conditional_losses_8856
A__inference_model_1_layer_call_and_return_conditional_losses_9557?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
'__inference_conv3d_3_layer_call_fn_9884?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
B__inference_conv3d_3_layer_call_and_return_conditional_losses_9875?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
.__inference_max_pooling3d_2_layer_call_fn_8266?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *M?J
H?EA?????????????????????????????????????????????
?2?
I__inference_max_pooling3d_2_layer_call_and_return_conditional_losses_8260?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *M?J
H?EA?????????????????????????????????????????????
?2?
'__inference_conv3d_4_layer_call_fn_9904?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
B__inference_conv3d_4_layer_call_and_return_conditional_losses_9895?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
.__inference_max_pooling3d_3_layer_call_fn_8278?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *M?J
H?EA?????????????????????????????????????????????
?2?
I__inference_max_pooling3d_3_layer_call_and_return_conditional_losses_8272?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *M?J
H?EA?????????????????????????????????????????????
?2?
)__inference_flatten_13_layer_call_fn_9915?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_flatten_13_layer_call_and_return_conditional_losses_9910?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
'__inference_dense_12_layer_call_fn_9935?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
B__inference_dense_12_layer_call_and_return_conditional_losses_9926?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
'__inference_dense_13_layer_call_fn_9955?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
B__inference_dense_13_layer_call_and_return_conditional_losses_9946?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
'__inference_dense_14_layer_call_fn_9975?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
B__inference_dense_14_layer_call_and_return_conditional_losses_9966?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
'__inference_dense_15_layer_call_fn_9995?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
B__inference_dense_15_layer_call_and_return_conditional_losses_9986?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
(__inference_dense_16_layer_call_fn_10015?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
C__inference_dense_16_layer_call_and_return_conditional_losses_10006?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
(__inference_dense_17_layer_call_fn_10035?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
C__inference_dense_17_layer_call_and_return_conditional_losses_10026?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
(__inference_dense_18_layer_call_fn_10055?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
C__inference_dense_18_layer_call_and_return_conditional_losses_10046?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
(__inference_dense_19_layer_call_fn_10075?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
C__inference_dense_19_layer_call_and_return_conditional_losses_10066?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
(__inference_dense_20_layer_call_fn_10095?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
C__inference_dense_20_layer_call_and_return_conditional_losses_10086?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
(__inference_dense_21_layer_call_fn_10115?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
C__inference_dense_21_layer_call_and_return_conditional_losses_10106?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
(__inference_dense_22_layer_call_fn_10135?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
C__inference_dense_22_layer_call_and_return_conditional_losses_10126?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
(__inference_dense_23_layer_call_fn_10155?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
C__inference_dense_23_layer_call_and_return_conditional_losses_10146?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
*__inference_flatten_14_layer_call_fn_10166?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
E__inference_flatten_14_layer_call_and_return_conditional_losses_10161?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
*__inference_flatten_15_layer_call_fn_10177?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
E__inference_flatten_15_layer_call_and_return_conditional_losses_10172?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
*__inference_flatten_16_layer_call_fn_10188?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
E__inference_flatten_16_layer_call_and_return_conditional_losses_10183?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
*__inference_flatten_17_layer_call_fn_10199?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
E__inference_flatten_17_layer_call_and_return_conditional_losses_10194?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
*__inference_flatten_18_layer_call_fn_10210?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
E__inference_flatten_18_layer_call_and_return_conditional_losses_10205?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
*__inference_flatten_19_layer_call_fn_10221?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
E__inference_flatten_19_layer_call_and_return_conditional_losses_10216?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
*__inference_flatten_20_layer_call_fn_10232?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
E__inference_flatten_20_layer_call_and_return_conditional_losses_10227?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
*__inference_flatten_21_layer_call_fn_10243?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
E__inference_flatten_21_layer_call_and_return_conditional_losses_10238?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
*__inference_flatten_22_layer_call_fn_10254?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
E__inference_flatten_22_layer_call_and_return_conditional_losses_10249?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
*__inference_flatten_23_layer_call_fn_10265?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
E__inference_flatten_23_layer_call_and_return_conditional_losses_10260?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
*__inference_flatten_24_layer_call_fn_10276?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
E__inference_flatten_24_layer_call_and_return_conditional_losses_10271?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
*__inference_flatten_25_layer_call_fn_10287?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
E__inference_flatten_25_layer_call_and_return_conditional_losses_10282?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
"__inference_signature_wrapper_9416input_2"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 ?
__inference__wrapped_model_8254?&'01??z{tunohibc\]VWPQJKDE>?<?9
2?/
-?*
input_2?????????ppd
? "???
2

flatten_14$?!

flatten_14?????????
2

flatten_15$?!

flatten_15?????????
2

flatten_16$?!

flatten_16?????????
2

flatten_17$?!

flatten_17?????????
2

flatten_18$?!

flatten_18?????????
2

flatten_19$?!

flatten_19?????????
2

flatten_20$?!

flatten_20?????????
2

flatten_21$?!

flatten_21?????????
2

flatten_22$?!

flatten_22?????????
2

flatten_23$?!

flatten_23?????????
2

flatten_24$?!

flatten_24?????????
2

flatten_25$?!

flatten_25??????????
B__inference_conv3d_3_layer_call_and_return_conditional_losses_9875t&';?8
1?.
,?)
inputs?????????ppd
? "1?.
'?$
0?????????66
? ?
'__inference_conv3d_3_layer_call_fn_9884g&';?8
1?.
,?)
inputs?????????ppd
? "$?!?????????66?
B__inference_conv3d_4_layer_call_and_return_conditional_losses_9895t01;?8
1?.
,?)
inputs?????????
? "1?.
'?$
0?????????

? ?
'__inference_conv3d_4_layer_call_fn_9904g01;?8
1?.
,?)
inputs?????????
? "$?!?????????
?
B__inference_dense_12_layer_call_and_return_conditional_losses_9926]>?0?-
&?#
!?
inputs??????????
? "%?"
?
0?????????
? {
'__inference_dense_12_layer_call_fn_9935P>?0?-
&?#
!?
inputs??????????
? "???????????
B__inference_dense_13_layer_call_and_return_conditional_losses_9946]DE0?-
&?#
!?
inputs??????????
? "%?"
?
0?????????
? {
'__inference_dense_13_layer_call_fn_9955PDE0?-
&?#
!?
inputs??????????
? "???????????
B__inference_dense_14_layer_call_and_return_conditional_losses_9966]JK0?-
&?#
!?
inputs??????????
? "%?"
?
0?????????
? {
'__inference_dense_14_layer_call_fn_9975PJK0?-
&?#
!?
inputs??????????
? "???????????
B__inference_dense_15_layer_call_and_return_conditional_losses_9986]PQ0?-
&?#
!?
inputs??????????
? "%?"
?
0?????????
? {
'__inference_dense_15_layer_call_fn_9995PPQ0?-
&?#
!?
inputs??????????
? "???????????
C__inference_dense_16_layer_call_and_return_conditional_losses_10006]VW0?-
&?#
!?
inputs??????????
? "%?"
?
0?????????
? |
(__inference_dense_16_layer_call_fn_10015PVW0?-
&?#
!?
inputs??????????
? "???????????
C__inference_dense_17_layer_call_and_return_conditional_losses_10026]\]0?-
&?#
!?
inputs??????????
? "%?"
?
0?????????
? |
(__inference_dense_17_layer_call_fn_10035P\]0?-
&?#
!?
inputs??????????
? "???????????
C__inference_dense_18_layer_call_and_return_conditional_losses_10046]bc0?-
&?#
!?
inputs??????????
? "%?"
?
0?????????
? |
(__inference_dense_18_layer_call_fn_10055Pbc0?-
&?#
!?
inputs??????????
? "???????????
C__inference_dense_19_layer_call_and_return_conditional_losses_10066]hi0?-
&?#
!?
inputs??????????
? "%?"
?
0?????????
? |
(__inference_dense_19_layer_call_fn_10075Phi0?-
&?#
!?
inputs??????????
? "???????????
C__inference_dense_20_layer_call_and_return_conditional_losses_10086]no0?-
&?#
!?
inputs??????????
? "%?"
?
0?????????
? |
(__inference_dense_20_layer_call_fn_10095Pno0?-
&?#
!?
inputs??????????
? "???????????
C__inference_dense_21_layer_call_and_return_conditional_losses_10106]tu0?-
&?#
!?
inputs??????????
? "%?"
?
0?????????
? |
(__inference_dense_21_layer_call_fn_10115Ptu0?-
&?#
!?
inputs??????????
? "???????????
C__inference_dense_22_layer_call_and_return_conditional_losses_10126]z{0?-
&?#
!?
inputs??????????
? "%?"
?
0?????????
? |
(__inference_dense_22_layer_call_fn_10135Pz{0?-
&?#
!?
inputs??????????
? "???????????
C__inference_dense_23_layer_call_and_return_conditional_losses_10146_??0?-
&?#
!?
inputs??????????
? "%?"
?
0?????????
? ~
(__inference_dense_23_layer_call_fn_10155R??0?-
&?#
!?
inputs??????????
? "???????????
D__inference_flatten_13_layer_call_and_return_conditional_losses_9910e;?8
1?.
,?)
inputs?????????

? "&?#
?
0??????????
? ?
)__inference_flatten_13_layer_call_fn_9915X;?8
1?.
,?)
inputs?????????

? "????????????
E__inference_flatten_14_layer_call_and_return_conditional_losses_10161X/?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????
? y
*__inference_flatten_14_layer_call_fn_10166K/?,
%?"
 ?
inputs?????????
? "???????????
E__inference_flatten_15_layer_call_and_return_conditional_losses_10172X/?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????
? y
*__inference_flatten_15_layer_call_fn_10177K/?,
%?"
 ?
inputs?????????
? "???????????
E__inference_flatten_16_layer_call_and_return_conditional_losses_10183X/?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????
? y
*__inference_flatten_16_layer_call_fn_10188K/?,
%?"
 ?
inputs?????????
? "???????????
E__inference_flatten_17_layer_call_and_return_conditional_losses_10194X/?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????
? y
*__inference_flatten_17_layer_call_fn_10199K/?,
%?"
 ?
inputs?????????
? "???????????
E__inference_flatten_18_layer_call_and_return_conditional_losses_10205X/?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????
? y
*__inference_flatten_18_layer_call_fn_10210K/?,
%?"
 ?
inputs?????????
? "???????????
E__inference_flatten_19_layer_call_and_return_conditional_losses_10216X/?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????
? y
*__inference_flatten_19_layer_call_fn_10221K/?,
%?"
 ?
inputs?????????
? "???????????
E__inference_flatten_20_layer_call_and_return_conditional_losses_10227X/?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????
? y
*__inference_flatten_20_layer_call_fn_10232K/?,
%?"
 ?
inputs?????????
? "???????????
E__inference_flatten_21_layer_call_and_return_conditional_losses_10238X/?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????
? y
*__inference_flatten_21_layer_call_fn_10243K/?,
%?"
 ?
inputs?????????
? "???????????
E__inference_flatten_22_layer_call_and_return_conditional_losses_10249X/?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????
? y
*__inference_flatten_22_layer_call_fn_10254K/?,
%?"
 ?
inputs?????????
? "???????????
E__inference_flatten_23_layer_call_and_return_conditional_losses_10260X/?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????
? y
*__inference_flatten_23_layer_call_fn_10265K/?,
%?"
 ?
inputs?????????
? "???????????
E__inference_flatten_24_layer_call_and_return_conditional_losses_10271X/?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????
? y
*__inference_flatten_24_layer_call_fn_10276K/?,
%?"
 ?
inputs?????????
? "???????????
E__inference_flatten_25_layer_call_and_return_conditional_losses_10282X/?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????
? y
*__inference_flatten_25_layer_call_fn_10287K/?,
%?"
 ?
inputs?????????
? "???????????
I__inference_max_pooling3d_2_layer_call_and_return_conditional_losses_8260?_?\
U?R
P?M
inputsA?????????????????????????????????????????????
? "U?R
K?H
0A?????????????????????????????????????????????
? ?
.__inference_max_pooling3d_2_layer_call_fn_8266?_?\
U?R
P?M
inputsA?????????????????????????????????????????????
? "H?EA??????????????????????????????????????????????
I__inference_max_pooling3d_3_layer_call_and_return_conditional_losses_8272?_?\
U?R
P?M
inputsA?????????????????????????????????????????????
? "U?R
K?H
0A?????????????????????????????????????????????
? ?
.__inference_max_pooling3d_3_layer_call_fn_8278?_?\
U?R
P?M
inputsA?????????????????????????????????????????????
? "H?EA??????????????????????????????????????????????
A__inference_model_1_layer_call_and_return_conditional_losses_8856?&'01??z{tunohibc\]VWPQJKDE>?D?A
:?7
-?*
input_2?????????ppd
p

 
? "???
???
?
0/0?????????
?
0/1?????????
?
0/2?????????
?
0/3?????????
?
0/4?????????
?
0/5?????????
?
0/6?????????
?
0/7?????????
?
0/8?????????
?
0/9?????????
?
0/10?????????
?
0/11?????????
? ?
A__inference_model_1_layer_call_and_return_conditional_losses_8956?&'01??z{tunohibc\]VWPQJKDE>?D?A
:?7
-?*
input_2?????????ppd
p 

 
? "???
???
?
0/0?????????
?
0/1?????????
?
0/2?????????
?
0/3?????????
?
0/4?????????
?
0/5?????????
?
0/6?????????
?
0/7?????????
?
0/8?????????
?
0/9?????????
?
0/10?????????
?
0/11?????????
? ?
A__inference_model_1_layer_call_and_return_conditional_losses_9557?&'01??z{tunohibc\]VWPQJKDE>?C?@
9?6
,?)
inputs?????????ppd
p

 
? "???
???
?
0/0?????????
?
0/1?????????
?
0/2?????????
?
0/3?????????
?
0/4?????????
?
0/5?????????
?
0/6?????????
?
0/7?????????
?
0/8?????????
?
0/9?????????
?
0/10?????????
?
0/11?????????
? ?
A__inference_model_1_layer_call_and_return_conditional_losses_9698?&'01??z{tunohibc\]VWPQJKDE>?C?@
9?6
,?)
inputs?????????ppd
p 

 
? "???
???
?
0/0?????????
?
0/1?????????
?
0/2?????????
?
0/3?????????
?
0/4?????????
?
0/5?????????
?
0/6?????????
?
0/7?????????
?
0/8?????????
?
0/9?????????
?
0/10?????????
?
0/11?????????
? ?
&__inference_model_1_layer_call_fn_9140?&'01??z{tunohibc\]VWPQJKDE>?D?A
:?7
-?*
input_2?????????ppd
p

 
? "???
?
0?????????
?
1?????????
?
2?????????
?
3?????????
?
4?????????
?
5?????????
?
6?????????
?
7?????????
?
8?????????
?
9?????????
?
10?????????
?
11??????????
&__inference_model_1_layer_call_fn_9323?&'01??z{tunohibc\]VWPQJKDE>?D?A
:?7
-?*
input_2?????????ppd
p 

 
? "???
?
0?????????
?
1?????????
?
2?????????
?
3?????????
?
4?????????
?
5?????????
?
6?????????
?
7?????????
?
8?????????
?
9?????????
?
10?????????
?
11??????????
&__inference_model_1_layer_call_fn_9781?&'01??z{tunohibc\]VWPQJKDE>?C?@
9?6
,?)
inputs?????????ppd
p

 
? "???
?
0?????????
?
1?????????
?
2?????????
?
3?????????
?
4?????????
?
5?????????
?
6?????????
?
7?????????
?
8?????????
?
9?????????
?
10?????????
?
11??????????
&__inference_model_1_layer_call_fn_9864?&'01??z{tunohibc\]VWPQJKDE>?C?@
9?6
,?)
inputs?????????ppd
p 

 
? "???
?
0?????????
?
1?????????
?
2?????????
?
3?????????
?
4?????????
?
5?????????
?
6?????????
?
7?????????
?
8?????????
?
9?????????
?
10?????????
?
11??????????
"__inference_signature_wrapper_9416?&'01??z{tunohibc\]VWPQJKDE>?G?D
? 
=?:
8
input_2-?*
input_2?????????ppd"???
2

flatten_14$?!

flatten_14?????????
2

flatten_15$?!

flatten_15?????????
2

flatten_16$?!

flatten_16?????????
2

flatten_17$?!

flatten_17?????????
2

flatten_18$?!

flatten_18?????????
2

flatten_19$?!

flatten_19?????????
2

flatten_20$?!

flatten_20?????????
2

flatten_21$?!

flatten_21?????????
2

flatten_22$?!

flatten_22?????????
2

flatten_23$?!

flatten_23?????????
2

flatten_24$?!

flatten_24?????????
2

flatten_25$?!

flatten_25?????????