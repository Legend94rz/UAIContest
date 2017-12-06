function varargout = UI(varargin)
% UI MATLAB code for UI.fig
%      UI, by itself, creates a new UI or raises the existing
%      singleton*.
%
%      H = UI returns the handle to a new UI or the handle to
%      the existing singleton*.
%
%      UI('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in UI.M with the given input arguments.
%
%      UI('Property','Value',...) creates a new UI or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before UI_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to UI_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help UI

% Last Modified by GUIDE v2.5 30-Nov-2017 18:35:08

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @UI_OpeningFcn, ...
                   'gui_OutputFcn',  @UI_OutputFcn, ...
                   'gui_LayoutFcn',  [] , ...
                   'gui_Callback',   []);
if nargin && ischar(varargin{1})
    gui_State.gui_Callback = str2func(varargin{1});
end

if nargout
    [varargout{1:nargout}] = gui_mainfcn(gui_State, varargin{:});
else
    gui_mainfcn(gui_State, varargin{:});
end
% End initialization code - DO NOT EDIT


% --- Executes just before UI is made visible.
function UI_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to UI (see VARARGIN)

% Choose default command line output for UI
handles.output = hObject;

% Update handles structure
guidata(hObject, handles);

% UIWAIT makes UI wait for user response (see UIRESUME)
% uiwait(handles.figure1);
global Fe
load('Fe.mat','Fe');
global withoffset
load('withoffset.mat','withoffset');

% --- Outputs from this function are returned to the command line.
function varargout = UI_OutputFcn(hObject, eventdata, handles) 
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;


% --- Executes on slider movement.
function slider1_Callback(hObject, eventdata, handles)
% hObject    handle to slider1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'Value') returns position of slider
%        get(hObject,'Min') and get(hObject,'Max') to determine range of slider
	calc();

% --- Executes during object creation, after setting all properties.
function slider1_CreateFcn(hObject, eventdata, handles)
% hObject    handle to slider1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: slider controls usually have a light gray background.
if isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor',[.9 .9 .9]);
end
global objS1
objS1 = hObject;


% --- Executes on slider movement.
function slider2_Callback(hObject, eventdata, handles)
% hObject    handle to slider2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'Value') returns position of slider
%        get(hObject,'Min') and get(hObject,'Max') to determine range of slider
	calc();

% --- Executes during object creation, after setting all properties.
function slider2_CreateFcn(hObject, eventdata, handles)
% hObject    handle to slider2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: slider controls usually have a light gray background.
if isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor',[.9 .9 .9]);
end
global objS2
objS2 = hObject;

% --- Executes on slider movement.
function slider3_Callback(hObject, eventdata, handles)
% hObject    handle to slider3 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'Value') returns position of slider
%        get(hObject,'Min') and get(hObject,'Max') to determine range of slider
	calc();
	
	
% --- Executes during object creation, after setting all properties.
function slider3_CreateFcn(hObject, eventdata, handles)
% hObject    handle to slider3 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: slider controls usually have a light gray background.
if isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor',[.9 .9 .9]);
end
global objS3
objS3 = hObject;



% --- Executes on slider movement.
function slider4_Callback(hObject, eventdata, handles)
% hObject    handle to slider5 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'Value') returns position of slider
%        get(hObject,'Min') and get(hObject,'Max') to determine range of slider
	calc();

% --- Executes during object creation, after setting all properties.
function slider4_CreateFcn(hObject, eventdata, handles)
% hObject    handle to slider6 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: slider controls usually have a light gray background.
if isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor',[.9 .9 .9]);
end
global objS4
objS4 = hObject;

function [m,g] = getConfidence(pre,nxt,L,R)
	g = zeros(size(pre));
	t = (R-L+1);
	m = (pre+nxt)./t;
	g(t<8) = -0.1*t(t<8)+1;
	g(t>=8) = min(0.1*t(t>=8)-0.6,0.8);

function r = work(F)
	[m,g] = getConfidence(F{:,'pre'},F{:,'nxt'},F{:,'L'},F{:,'R'});
	r = m.*g + F{:,'meanOfHis'}.*(1-g);
		
	
function calc()
	global objS1
	global objS2
	global objS3
	global objS4
	global Fe
	global withoffset
	copy = withoffset;
	s = [ get(objS1,'Value'),get(objS2,'Value'),get(objS3,'Value'), get(objS4,'Value')]
	minD = 999999999;
	optoff = 0;
	for off = 1
		ind = withoffset.offset==off;
		withoffset.count(ind & (withoffset.count>1) ) = withoffset.count(ind & (withoffset.count>1) )*3;
		
		ind1 = ind&(withoffset.count==1);
		withoffset.count(ind1) = withoffset.count(ind1)*5;
		
		ind0 = ind&(withoffset.count==0);
		withoffset.count(ind0) = withoffset.count(ind0)+3;
		
		f = histcounts(withoffset.count,'BinWidth',1);
		tmp = abs(sum(f(1:3))-3272);
		if( tmp < minD)
			minD = tmp;
			optoff=off;
			cnt = f;
		end
		fprintf('off: %d, diff: %d, mean: %.2f\n',off,tmp,mean(withoffset.count));
		histogram(withoffset.count,'BinWidth',1);
		withoffset = copy;
	end
	
	
	%{
	test_id = (0:4999)';
	count = zeros(5000,1);
	s = [ get(objS1,'Value'),get(objS2,'Value'),get(objS3,'Value'), get(objS4,'Value')]
	ind = (Fe.sim>s(1))&( Fe.meanOfAll > 1) & (Fe.meanOfHis>=3);
	vind =~(ind);
	count(ind) = Fe.meanOfWeek(ind);
	
	count(vind) = work( Fe(vind,:) );	
	
	count = round(count+s(4));
	f = histogram(count,'BinWidth',1);
	res = table(test_id,count);
	fprintf('mean = %f\n', mean(count));
	fprintf('count012: %d\n', sum(f.Values(1:3)));
	%}
	%save('res.mat','res');
