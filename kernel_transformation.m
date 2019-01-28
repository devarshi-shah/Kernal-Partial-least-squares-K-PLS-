function K = kernel_transformation(X1,X2,ktype,kpar)
% function calculates the kernel matrix between two data sets.
% Input:	- X1, X2: data matrices in row format (data as rows)
%			- ktype: string representing kernel type
%			- kpar: vector containing the kernel parameters
% Output:	- K: kernel matrix
% USAGE: K = km_kernel(X1,X2,ktype,kpar)
%


switch ktype
	case 'gauss'	% Gaussian kernel
		sgm = kpar;	% kernel width
		
		dim1 = size(X1,1);
		dim2 = size(X2,1);
		
		norms1 = sum(X1.^2,2);
		norms2 = sum(X2.^2,2);
		
		mat1 = repmat(norms1,1,dim2);
		mat2 = repmat(norms2',dim1,1);
		
		distmat = mat1 + mat2 - 2*X1*X2';	% full distance matrix
		K = exp(-distmat/(2*sgm^2));
		
	case 'gauss-diag'	% only diagonal of Gaussian kernel
		sgm = kpar;	% kernel width
		K = exp(-sum((X1-X2).^2,2)/(2*sgm^2));
		
	case 'poly'	% polynomial kernel
		p = kpar(1);	% polynome order
		c = kpar(2);	% additive constant
		
		K = (X1*X2' + c).^p;
		
	case 'linear' % linear kernel
		K = X1*X2';
    
    case 'htan' %hyperbolic tangent kernel
        kpa=kpar(1);      % p >0 
        c=kpar(2);      % c<0
        K=tanh(kpa*X1*X2'+c);
    case 'lrbf'     %laplace radial basis function
        sgm=kpar;
        l1=size(X1,1);
        l2=size(X2,1);
        K=zeros(l1,l2);
        for i=1:l1
            for j=1:l2
                K(i,j)=exp(-(norm(X1(i,:)-X2(j,:)))/sgm);
                
            end
        end
        %K=exp(-
	otherwise	% default case
		error ('unknown kernel type')
end
