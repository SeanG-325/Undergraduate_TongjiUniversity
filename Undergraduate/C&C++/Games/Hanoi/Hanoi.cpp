/*1552192 管硕 数强 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <conio.h>
#include <windows.h>
#include <iostream>
#include <time.h>
/*为方便修改而设定的若干宏定义*/ 
#define MIN_HEIGHT 21 //最小高度 
#define MAX_HEIGHT 10 //最大高度 
#define MOVE_HEIGHT 8 //移动高度 
#define LEFT_BOARD 5 //左边界 
#define RIGHT_BOARD 88 //右边界 
#define LEFT_ROD 16 //左杆水平位置 
#define MID_ROD 46 //中间杆水平位置 
#define RIGHT_ROD 76 //右杆水平位置 
using namespace std;
const int rod[3]={16,46,76}; //与杆位置对应 
const int bg_color = 14;
const int fg_color = 9;
const char ch = ' ';
int LENGTH = 23;
int length,step=0;
char a, b, c;
int i[100][11] = { 0 };
int index, check_1 = 0, icount = 0, j, k,d,l;
char get_in_1,get_in_2;
HANDLE hout = GetStdHandle(STD_OUTPUT_HANDLE);
POINT point;
void move(char A,char B);
void getxy();
void gotoxy(HANDLE hout, const int X, const int Y);
void setcolor(HANDLE hout, const int bg_color, const int fg_color);
int check5()
{
	int temp,icount[2]={0};
	for(temp=0;temp<10;temp++) 
	{
		if(i[a][temp]!=0) icount[0]++;
		if(i[c][temp]!=0) icount[1]++;
	}
	if(icount[0]==0&&icount[1]==index) return 1;
	else return 0;
}
int check4()
{
	int temp,top_start,top_end;
	for(temp=0;i[get_in_1][temp+1]!=0;temp++);
	top_start=i[get_in_1][temp];
	for(temp=0;i[get_in_2][temp+1]!=0;temp++);
	top_end=i[get_in_2][temp];
	if(top_end!=0&&top_end<=top_start||top_start==0) return 0;
	else return 1;
}
void get_in()
{
	while(1)
	{
		printf("Enter the starting tower name:");
		get_in_1=_getche();
		printf("\n");
		if(get_in_1==97||get_in_1==98||get_in_1==99)
		{
			get_in_1-=32;
		}
		if(!(get_in_1==65||get_in_1==66||get_in_1==67))
		{
			printf("Error!\n");
			getxy();
			continue;
		}
		printf("Enter the ending tower name:");
		get_in_2=_getche();
		printf("\n");
		if(get_in_2==97||get_in_2==98||get_in_2==99)
		{
			get_in_2-=32;
		}
		if(!(get_in_2==65||get_in_2==66||get_in_2==67)||get_in_2==get_in_1)
		{
			printf("Error!\n");
			getxy();
			continue;
		}
		if(!check4())
		{
			printf("Error!\n");
			getxy();
			continue;
		}
		else break;
    }
    move(get_in_1,get_in_2);
    getxy();
    gotoxy(hout,0,30);
    setcolor(hout, 0, 7);
    step++;
}
void sleep()
{
	if(d==0)
	{
		getchar();
		getchar();
	}
	else if(d==1) Sleep(300);
	else if(d==2) Sleep(200);
	else if(d==3) Sleep(100);
	else if(d==4) Sleep(1);
}
int check3()
{
	if (d < 0 || d > 5)
	{
		printf("Error!\n");
		return 0;
	}
	else return 1;
}
void gotoxy(HANDLE hout, const int X, const int Y)
{
	COORD coord;
	coord.X = X;
	coord.Y = Y;
	SetConsoleCursorPosition(hout, coord);
}
void setcolor(HANDLE hout, const int bg_color, const int fg_color)
{
	SetConsoleTextAttribute(hout, bg_color * 16 + fg_color);
}
void showch(HANDLE hout, const int X, const int Y, const int bg_color, const int fg_color, const char ch, const int n)
{
	int i;
	gotoxy(hout, X, Y);
	setcolor(hout, bg_color, fg_color);
	for (i = 0; i<n; i++)
		putchar(ch);
}
void error()
{
	cin.clear();
	cin.ignore(1024, '\n');
	cin.clear();
}
int check1()
{
	if (index <= 0 || index >= 11)
	{
		printf("Error!\n");
		return 0;
	}
	else return 1;
}
void check2()
{
	check_1 = 0;
	if (int(a) >= 97) a -= 32;
	if (int(b) >= 97) b -= 32;
	if (int(c) >= 97) c -= 32;
	if ((a == 'A') && (c == 'B'))      b = 'C';
	else if ((a == 'A') && (c == 'C')) b = 'B';
	else if ((a == 'B') && (c == 'C')) b = 'A';
	else if ((a == 'B') && (c == 'A')) b = 'C';
	else if ((a == 'C') && (c == 'A')) b = 'B';
	else if ((a == 'C') && (c == 'B')) b = 'A';
	else
	{
		printf("Error!\n");
		check_1 = 1;
	}
}
void initialize()
{
	int temp;
	for (temp = 0; temp<index; temp++) i[int(a)][temp] = index - temp;
	HANDLE hout = GetStdHandle(STD_OUTPUT_HANDLE);
	int temp_1;
	showch(hout,LEFT_BOARD   ,MIN_HEIGHT,bg_color,fg_color,ch,LENGTH);
	showch(hout,LEFT_BOARD+30,MIN_HEIGHT,bg_color,fg_color,ch,LENGTH);
	showch(hout,LEFT_BOARD+60,MIN_HEIGHT,bg_color,fg_color,ch,LENGTH);
	for(temp_1=1;temp_1<12;temp_1++)
	{
		showch(hout,LEFT_ROD ,MIN_HEIGHT-temp_1,bg_color,fg_color,ch,1);
    	showch(hout,MID_ROD  ,MIN_HEIGHT-temp_1,bg_color,fg_color,ch,1);
    	showch(hout,RIGHT_ROD,MIN_HEIGHT-temp_1,bg_color,fg_color,ch,1);
	}
	setcolor(hout, 0, 7); 
	for(temp=0;temp<10;temp++)
	{
		if(i[int(a)][temp]!=0)  showch(hout,LEFT_ROD+30*(int(a)-'A')-i[int(a)][temp],MIN_HEIGHT-temp-1,i[int(a)][temp]+1,2+temp,ch,i[int(a)][temp]*2+1);
	}
}
void move(char A,char B)
{
	int up=0,down=0,left=0,right=0,color=0,start_rod,end_rod,temp;
	for (j = 0; i[int(A)][j + 1] != 0; j++);
	l = i[int(A)][j];
	color=1+l;
	start_rod=rod[int(A)-65];//读取起始杆 
	end_rod  =rod[int(B)-65];//读取终止杆 
	for(temp=0;temp<12-j;temp++)
	{
		showch(hout,start_rod-l,MIN_HEIGHT-j-temp-1,color,fg_color,ch,2*l+1);//循环打印，宽度为盘子序号*2-1 
		sleep();
		if(MIN_HEIGHT-temp-j-1>MOVE_HEIGHT)
		{
			showch(hout,start_rod-l,MIN_HEIGHT-j-temp-1,0,7,ch,2*l+1); //控制消去起始点与宽度，与打印时一致 
			if(MIN_HEIGHT-temp-j-1>MOVE_HEIGHT+1) showch(hout,start_rod,MIN_HEIGHT-j-temp-1,bg_color,7,ch,1);//使最后一个不打印杆的颜色 
		}
    }
    i[int(A)][j] = 0;
    if(int(A)>int(B))//左移函数 
    {
    	left=30*(int(A)-int(B));
    	for(temp=0;temp<=left;temp++)
    	{
    		showch(hout,start_rod-l-temp,MOVE_HEIGHT,color,fg_color,ch,2*l+1);
    		sleep();
    		if(temp<=(int(B)-63)*30)
    		{
    			showch(hout,start_rod-l-temp,MOVE_HEIGHT,0,7,ch,2*l+1);
    		}
    	}
    }
    else//右移函数 
    {
    	right=30*(int(B)-int(A));
    	for(temp=0;temp<=right;temp++)
    	{
    		showch(hout,start_rod-l+temp,MOVE_HEIGHT,color,fg_color,ch,2*l+1);
    		sleep();
    		if(temp<=(int(B)-65)*30)
    		{
    			showch(hout,start_rod-l+temp,MOVE_HEIGHT,0,7,ch,2*l+1);
    		}
    	}
    }
   	for (k = 0; i[int(B)][k] != 0; k++);
	i[int(B)][k] = l;
	for(temp=1;temp<12-k+1;temp++)
	{
		showch(hout,end_rod-l,MOVE_HEIGHT+temp,color,fg_color,ch,2*l+1);
		sleep();
		if(MOVE_HEIGHT+temp-k<MIN_HEIGHT-1)
		{
			if(temp<12-k) showch(hout,end_rod-l,MOVE_HEIGHT+temp,0,7,ch,2*l+1);
			if(temp!=1&&temp<12-k) showch(hout,end_rod,MOVE_HEIGHT+temp,bg_color,7,ch,1);
		}
	}
}
void hanoi(int index, char A, char B, char C)
{
	if (index == 1) move(A, C);
	else if (index >= 2)
	{
		hanoi(index - 1, A, C, B);
		move(A, C);
		hanoi(index - 1, B, A, C);
	}
}
void getxy()
{
	Sleep(500);
	int temp;
	GetCursorPos(&point);
	if(point.y>35) 
	{
		for(temp=0;temp<10;temp++)
		showch(hout,0 ,30+temp,0,0,ch,40);
		gotoxy(hout,0,30);
		setcolor(hout, 0, 7);
	}
}
int main()
{
	system("mode con cols=100 lines=40");
	while(1)
	{
		printf("Enter the number(1-10):");
    	cin >> index;
    	if (!check1())
    	{
    		error();
    		continue;
    	}
    	printf("Enter the starting tower name(A-C):");
    	cin >> a;
    	error();
    	printf("Enter the ending tower name(A-C):");
		cin >> c;
		error();
    	check2();
    	printf("Enter the speed of print(0-5)(enter 0 to step by step):");
		cin>>d;
		error();
    	if(check_1==0&&check3()) break;
	}
	initialize();
	setcolor(hout, 0, 7); 
	gotoxy(hout,0,30);
	while(!check5())
	{
		get_in();
		setcolor(hout, 0, 7);
		gotoxy(hout,0,25);
		printf("\t\t\t\t\t    Step=%d",step);
		gotoxy(hout,0,30);
	}
	printf("\t\t\t\t\t   You win!\n");
	gotoxy(hout,0,35);//避免最终提示干扰输出信息 
	return 0;
}
