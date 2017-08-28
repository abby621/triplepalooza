clear; close all;

% we have a dim x n feature matrix, where n%k = 0 and each chunk of k
% features are from the same class
all_feats = rand(200,100);
all_feats(:,11:20) = all_feats(:,11:20)*2;
all_feats(:,21:30) = all_feats(:,21:30)*3;
all_feats(:,31:40) = all_feats(:,31:40)*4;
all_feats(:,41:50) = all_feats(:,41:50)*5;
all_feats(:,51:60) = all_feats(:,51:60)*6;
all_feats(:,61:70) = all_feats(:,61:70)*7;
all_feats(:,71:80) = all_feats(:,71:80)*8;
all_feats(:,81:90) = all_feats(:,81:90)*9;
all_feats(:,91:100) = all_feats(:,91:100)*10;

D = dist(all_feats);

margin = 100;

%%

% which "block" are our positive features in?
posIdx = floor(((1:100)-1)/10);

% where do the features in this block start?
posIdx10 = 10*posIdx;

% what are the positive feature indices?
posImInds = repmat(posIdx10,[10 1])'+repmat(1:10',[100 1]);

% what are our anchor images?
anchorInds = repmat(1:100,[10 1])';

% what are the positive feature indices?
posInds = sub2ind(size(D),anchorInds,posImInds);

% what are our positive distances?
posDists = D(posInds);
posDistsRep = repmat(shiftdim(posDists,-1),[100 1]);

% what are the negative feature indices?
allDists = repmat(D,[1 1 10]);

relDists = max(0,margin+posDistsRep-allDists);

%% make our mask
% negative anchor positive
[ra rb rc] = ndgrid(1:100,1:100,1:10);  % make 100,100,10 grid coordinates

% find the ones that are invalid triplets
diagonal = (floor((ra-1)./10)==floor((rb-1)./10)) & (floor(1+(ra-1)./10) == rc);

% TODO: we also want to mask when the positive == anchor and when the
% negative is in the positive range
bad_negatives = floor((ra-1)./10) == floor((rb-1)./10);
bad_positives = mod(rb-1,10) == mod(rc-1,10);

% make a mask where those are zero.
mask = (1-bad_negatives).*(1-bad_positives);

masked_relDists = mask.*relDists;

%%
close all; figure();
for anchorIm = 1:100
    clf;
    imagesc(squeeze(masked_relDists(:,anchorIm,:)));
    title(anchorIm);
    xlabel('Positive image');
    ylabel('Negative image');
    pause(.1);
end