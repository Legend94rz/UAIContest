C = zeros(10,10);
for i = 1:10
	for j = 1:10
		C(i,j) = A(2,i)-A(1,j);
	end
end
C(C<0)=0;