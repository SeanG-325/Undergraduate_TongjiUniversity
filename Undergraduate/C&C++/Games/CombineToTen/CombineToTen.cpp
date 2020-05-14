//1552192 数强 管硕 
#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <conio.h>
#include <windows.h>
#include <iostream>
#define SIZE 8
#define SIZE_1 5
using namespace std;
HANDLE hout = GetStdHandle(STD_OUTPUT_HANDLE);
void PrintUnit(int i, int X, int Y);
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
void finding_1(int(*i)[12], int(*temp)[12], int x, int y)
{
	*(*(temp + x) + y) = *(*(i + x) + y);
	if (*(*(i + x + 1) + y) != 0 && *(*(i + x + 1) + y) == *(*(i + x) + y) && *(*(temp + x + 1) + y) == 0) finding_1(i, temp, x + 1, y);
	if (*(*(i + x - 1) + y) != 0 && *(*(i + x - 1) + y) == *(*(i + x) + y) && *(*(temp + x - 1) + y) == 0) finding_1(i, temp, x - 1, y);
	if (*(*(i + x) + y + 1) != 0 && *(*(i + x) + y + 1) == *(*(i + x) + y) && *(*(temp + x) + y + 1) == 0) finding_1(i, temp, x, y + 1);
	if (*(*(i + x) + y - 1) != 0 && *(*(i + x) + y - 1) == *(*(i + x) + y) && *(*(temp + x) + y - 1) == 0) finding_1(i, temp, x, y - 1);
}
void error()
{
	cin.clear();
	cin.ignore(1024, '\n');
	cin.clear();
}
void windowsize_change(int x, int y)
{
	char change[100];
	sprintf(change, "mode con cols=%d lines=%d", 34 + SIZE*(x - 4), 25 + 4 * (y - 4));
	system(change);
}
void initialize_1(int(*i)[12], int x_1, int y_1)
{
	srand((unsigned int)(time(0)));
	int j, k;
	for (j = 1; j<x_1 + 1; j++)
		for (k = 1; k<y_1 + 1; k++)
			*(*(i + j) + k) = rand() % 3 + 1;
}
void initialize_2(int(*i)[12], int(*temp)[12], int max, int x_1, int y_1)
{
	int j, k;
	srand((unsigned int)(time(0)));
	if (max == 4)
	{
		for (j = 1; j<x_1 + 1; j++)
			for (k = 1; k<y_1 + 1; k++)
				if (*(*(i + j) + k) == 0)
				{
					if (rand() % 10 == 0) *(*(i + j) + k) = max;
					else if (rand() % 10 == 1 || rand() % 10 == 2 || rand() % 10 == 3) *(*(i + j) + k) = 1;
					else if (rand() % 10 == 4 || rand() % 10 == 5 || rand() % 10 == 6) *(*(i + j) + k) = 2;
					else if (rand() % 10 == 7 || rand() % 10 == 8 || rand() % 10 == 9) *(*(i + j) + k) = 3;
					else *(*(i + j) + k) = 1;
				}
	}
	else if (max == 5)
	{
		for (j = 1; j<x_1 + 1; j++)
			for (k = 1; k<y_1 + 1; k++)
				if (*(*(i + j) + k) == 0)
				{
					if (rand() % 4 == 0 && rand() % 2 == 0) *(*(i + j) + k) = max;
					else if (rand() % 4 == 0 && rand() % 2 == 1) *(*(i + j) + k) = 4;
					else if (rand() % 4 == 1) *(*(i + j) + k) = 1;
					else if (rand() % 4 == 2) *(*(i + j) + k) = 2;
					else if (rand() % 4 == 3) *(*(i + j) + k) = 3;
					else *(*(i + j) + k) = 1;
				}
	}
	else if (max == 6)
	{
		for (j = 1; j<x_1 + 1; j++)
			for (k = 1; k<y_1 + 1; k++)
				if (*(*(i + j) + k) == 0)
				{
					if (rand() % 5 == 0 && rand() % 4 == 0) *(*(i + j) + k) = max;
					else if (rand() % 5 == 0 && rand() % 4 != 0) *(*(i + j) + k) = 5;
					else if (rand() % 5 == 4) *(*(i + j) + k) = 4;
					else if (rand() % 5 == 1) *(*(i + j) + k) = 1;
					else if (rand() % 5 == 2) *(*(i + j) + k) = 2;
					else if (rand() % 5 == 3) *(*(i + j) + k) = 3;
					else *(*(i + j) + k) = 1;
				}
	}
	else if (max>6)
	{
		for (j = 1; j<x_1 + 1; j++)
			for (k = 1; k<y_1 + 1; k++)
				if (*(*(i + j) + k) == 0) *(*(i + j) + k) = rand() % max + 1;
	}
	else
	{
		for (j = 1; j<x_1 + 1; j++)
			for (k = 1; k<y_1 + 1; k++)
				if (*(*(i + j) + k) == 0) *(*(i + j) + k) = rand() % 3 + 1;
	}
}
int find_max(int(*i)[12])
{
	int max = 0;
	int j, k;
	for (j = 0; j<12; j++)
		for (k = 0; k<12; k++)
			if (max<*(*(i + j) + k)) max = *(*(i + j) + k);
	return max;
}
int iscore(int(*temp)[12])
{
	int j, k, l = 0, m = 0, n = 0;
	for (j = 0; j<12; j++)
		for (k = 0; k<12; k++)
			if (*(*(temp + j) + k)>0)
			{
				m = *(*(temp + j) + k);
				n++;
			}
	return m*n * 3;
}
int check(int(*i)[12], int(*temp)[12], int x_1, int y_1)
{
	int j, k, l = 0, m, n, p = 0, j_1, k_1;
	for (j = 1; j<x_1 + 1; j++)
	{
		for (k = 1; k<y_1 + 1; k++)
		{
			p = 0;
			finding_1(i, temp, j, k);
			for (m = 0; m<12; m++)
				for (n = 0; n<12; n++)
					if (*(*(temp + m) + n) != 0) p++;
			for (j_1 = 0; j_1<12; j_1++)
				for (k_1 = 0; k_1<12; k_1++)
					*(*(temp + j_1) + k_1) = 0;
			if (p != 1) l++;
		}
		if (l != 0) break;
		l = 0;
	}
	for (j = 0; j<12; j++)
		for (k = 0; k<12; k++)
			*(*(temp + j) + k) = 0;
	return l;
}
int getx()
{
	POINT point;
	GetCursorPos(&point);
	return point.x;
}
int gety()
{
	POINT point;
	GetCursorPos(&point);
	return point.y;
}
void background_lines(int x, int y)//╠ ╣ ╦ ╩ ╬
{
	int j, k;
	gotoxy(hout, 0, 1);
	setcolor(hout, 15, 0);
	printf("╔");
	for (j = 0; j<(35 + SIZE*(x - 4)) / 2 - 2; j++)
	{
		if (j % 4 != 3) printf("═");
		else printf("╦");
	}
	printf("╗");
	for (j = 0; j<(19 + 4 * (y - 4)) - 4; j++)
	{
		gotoxy(hout, 0, j + 2);
		setcolor(hout, 15, 0);
		if (j % 4 != 3) printf("║");
		else printf("╠");
		if (j % 4 != 3)
			for (k = 0; k<(35 + SIZE*(x - 4) - 3); k++)
			{
				if (k % 7 != 6) printf(" ");
				else printf("║");
			}
		else
			for (k = 0; k<(35 + SIZE*(x - 4) - 3) / 2; k++)
			{
				if (k % 4 != 3) printf("═");
				else printf("╬");
			}
		gotoxy(hout, (35 + SIZE*(x - 4)) + 2 - 5, j + 2);
		if (j % 4 != 3)
			printf("║");
		else
			printf("╣");
		setcolor(hout, 0, 7);
	}
	gotoxy(hout, 0, j + 2);
	setcolor(hout, 15, 0);
	printf("╚");
	for (j = 0; j<(35 + SIZE*(x - 4)) / 2 - 2; j++)
	{
		if (j % 4 != 3) printf("═");
		else printf("╩");
	}
	printf("╝");
}
void print(int(*i)[12], int x_1, int y_1)
{
	int j, k, y = 3, x = 5;
	gotoxy(hout, 0, 4);
	for (j = 1; j<x_1 + 1; j++)
	{
		for (k = 1; k<y_1 + 1; k++)
		{
			gotoxy(hout, x, y);
			PrintUnit(*(*(i + j) + k), x, y);
			y += 4;
		}
		printf("\n");
		x += 8;
		y = 3;
	}
}
void PrintUnit(int i, int X, int Y)
{
	if (i != 0)
	{
		setcolor(hout, i, 17 - i);
		gotoxy(hout, X - 3, Y - 1);
		printf("╔═╗");
		gotoxy(hout, X - 3, Y);
		printf("║%2d║", i);
		gotoxy(hout, X - 3, Y + 1);
		printf("╚═╝");
	}
	else if (i == 0)
	{
		setcolor(hout, 15, 0);
		gotoxy(hout, X - 3, Y - 1);
		printf("      ");
		gotoxy(hout, X - 3, Y);
		printf("      ");
		gotoxy(hout, X - 3, Y + 1);
		printf("      ");
	}
	else if (i == -1)
	{
		setcolor(hout, 15, 0);
		gotoxy(hout, X - 3, Y - 2);
		printf("═══");
	}
}
void PrintUnit_1(int i, int X, int Y)
{
	setcolor(hout, 14, 1);
	gotoxy(hout, X - 3, Y - 1);
	printf("╔═╗");
	gotoxy(hout, X - 3, Y);
	printf("║%2d║", i);
	gotoxy(hout, X - 3, Y + 1);
	printf("╚═╝");
}
void left(int(*i)[12], int(*temp)[12], int x_1, int y_1)
{
	int j, k;
	for (j = 1; j<x_1 + 1; j++)
	{
		for (k = 1; k<y_1 + 1; k++)
		{
			if (*(*(temp + j) + k) == 1) break;
		}
		if (*(*(temp + j) + k) == 1) break;
	}
	PrintUnit(*(*(i + j) + k), 8 * j - 3, 4 * k - 1);
	*(*(temp + j) + k) = 0;
	if (j - 1<1) j = x_1;
	else j -= 1;
	*(*(temp + j) + k) = 1;
}
void right(int(*i)[12], int(*temp)[12], int x_1, int y_1)
{
	int j, k;
	for (j = 1; j<x_1 + 1; j++)
	{
		for (k = 1; k<y_1 + 1; k++)
		{
			if (*(*(temp + j) + k) == 1) break;
		}
		if (*(*(temp + j) + k) == 1) break;
	}
	PrintUnit(*(*(i + j) + k), 8 * j - 3, 4 * k - 1);
	*(*(temp + j) + k) = 0;
	if (j + 1>x_1) j = 1;
	else j += 1;
	*(*(temp + j) + k) = 1;
}
void up(int(*i)[12], int(*temp)[12], int x_1, int y_1)
{
	int j, k;
	for (j = 1; j<x_1 + 1; j++)
	{
		for (k = 1; k<y_1 + 1; k++)
		{
			if (*(*(temp + j) + k) == 1) break;
		}
		if (*(*(temp + j) + k) == 1) break;
	}
	PrintUnit(*(*(i + j) + k), 8 * j - 3, 4 * k - 1);
	*(*(temp + j) + k) = 0;
	if (k - 1<1) k = y_1;
	else k -= 1;
	*(*(temp + j) + k) = 1;
}
void down(int(*i)[12], int(*temp)[12], int x_1, int y_1)
{
	int j, k;
	for (j = 1; j<x_1 + 1; j++)
	{
		for (k = 1; k<y_1 + 1; k++)
		{
			if (*(*(temp + j) + k) == 1) break;
		}
		if (*(*(temp + j) + k) == 1) break;
	}
	PrintUnit(*(*(i + j) + k), 8 * j - 3, 4 * k - 1);
	*(*(temp + j) + k) = 0;
	if (k + 1>y_1) k = 1;
	else k += 1;
	*(*(temp + j) + k) = 1;
}
void RebuildTempArray(int(*temp)[12])
{
	int j, k;
	for (j = 0; j<12; j++)
		for (k = 0; k<12; k++)
			*(*(temp + j) + k) = 0;
}
void MoveUnit(int(*i)[12], int x_1, int y_1)
{
	int j, k, l;
	for (j = 1; j<x_1 + 1; j++)
		for (k = y_1 + 1; k>0; k--)
			for (l = k; l<y_1; l++)
				if (*(*(i + j) + l + 1) == 0)
				{
					print(i, x_1, y_1);
					if (*(*(i + j) + l) == 0) continue;
					PrintUnit(0, 8 * j - 3, 4 * l - 1);
					PrintUnit(*(*(i + j) + l), 8 * j - 3, 4 * l);
					Sleep(50);
					PrintUnit(0, 8 * j - 3, 4 * l);
					PrintUnit(*(*(i + j) + l), 8 * j - 3, 4 * l + 1);
					Sleep(50);
					PrintUnit(0, 8 * j - 3, 4 * l + 1);
					PrintUnit(*(*(i + j) + l), 8 * j - 3, 4 * l + 2);
					Sleep(50);
					PrintUnit(0, 8 * j - 3, 4 * l + 2);
					PrintUnit(*(*(i + j) + l), 8 * j - 3, 4 * l + 3);
					*(*(i + j) + l + 1) = *(*(i + j) + l);
					*(*(i + j) + l) = 0;
				}
}
void PrintSelectUnit(int(*i)[12], int(*temp)[12], int x_1, int y_1)
{
	int j, k;
	for (j = 1; j<x_1 + 1; j++)
	{
		for (k = 1; k<y_1 + 1; k++)
		{
			if (*(*(temp + j) + k) != 0) break;
		}
		if (*(*(temp + j) + k) != 0) break;
	}
	gotoxy(hout, 8 * j - 2, 4 * k);
	PrintUnit_1(*(*(i + j) + k), 8 * j - 3, 4 * k - 1);
}
void PrintSearchUnit(int(*i)[12], int(*temp)[12], int x_1, int y_1)
{
	int j, k;
	for (j = 1; j<x_1 + 1; j++)
	{
		for (k = 1; k<y_1 + 1; k++)
		{
			if (*(*(temp + j) + k) == *(*(i + j) + k))
			{
				PrintUnit_1(*(*(i + j) + k), 8 * j - 3, 4 * k - 1);
			}
		}
	}
}
void CombineUnit(int(*i)[12], int(*temp)[12], int x, int y, int x_1, int y_1)
{
	int j, k, l;
	*(*(i + x) + y) += 1;
	l = *(*(i + x) + y);
	for (j = 1; j<x_1 + 1; j++)
	{
		for (k = 1; k<y_1 + 1; k++)
		{
			if (*(*(temp + j) + k) == l - 1) *(*(i + j) + k) = 0;
		}
	}
	*(*(i + x) + y) = l;
}
int CombineOrNot(int(*temp)[12])
{
	int j, k, icount = 0;
	for (j = 0; j<12; j++)
		for (k = 0; k<12; k++)
		{
			if (*(*(temp + j) + k) != 0) icount++;
		}
	if (icount == 1) return 0;
	else return 1;
}
void SearchUnit(int(*i)[12], int(*temp)[12]/*移动临时数组*/, int(*temp_1)[12]/*合并临时数组*/, int x_1, int y_1)
{
	int j, k, x, y;
	for (j = 1; j<x_1 + 1; j++)
	{
		for (k = 1; k<y_1 + 1; k++)
		{
			if (*(*(temp + j) + k) == 1) break;
		}
		if (*(*(temp + j) + k) == 1) break;
	}
	x = j;
	y = k;
	finding_1(i, temp_1, x, y);
	PrintSearchUnit(i, temp_1, x_1, y_1);
	gotoxy(hout, 0, 20 + 4 * (y_1 - 4) - 2);
	setcolor(hout, 0, 7);
	printf("Press Enter to combine,others to reselect:");
	j = _getch();
	if (j == 13 && CombineOrNot(temp_1))
	{
		CombineUnit(i, temp_1, x, y, x_1, y_1);
		MoveUnit(i, x_1, y_1);
	}
	else
	{
		print(i, x_1, y_1);
		RebuildTempArray(temp_1);
		return;
	}
}
void SelectUnit(int(*i)[12], int(*temp)[12], int(*temp_1)[12], int x_1, int y_1)
{
	char key;
	gotoxy(hout, 0, 20 + 4 * (y_1 - 4) - 2);
	setcolor(hout, 0, 7);
	printf("Use up/down/left/right to control:");
	PrintSelectUnit(i, temp, x_1, y_1);
	while (1)
	{
		key = _getch();
		if (key == 72)      up(i, temp, x_1, y_1);
		else if (key == 80) down(i, temp, x_1, y_1);
		else if (key == 75) left(i, temp, x_1, y_1);
		else if (key == 77) right(i, temp, x_1, y_1);
		else if (key == 13)
		{
			SearchUnit(i, temp, temp_1, x_1, y_1);
			print(i, x_1, y_1);
		}
		else continue;
		break;
	}
}
int main()
{
	int index[12][12] = { 0 }, temp[12][12] = { 0 }, temp_1[12][12] = { 0 }, temp_2[12][12] = { 0 };
	int(*i)[12] = index;
	int(*i_1)[12] = temp;
	int(*i_temp)[12] = temp_1;
	int(*i_temp_2)[12] = temp_2;
	temp[1][1] = 1;
	int x_1, y_1;
	int max, value = 0, score = 0;
	while (1)
	{
		printf("Enter the max of x(5-8):");
		cin >> x_1;
		error();
		if (!(x_1 >= 5 && x_1 <= 8))
		{
			printf("Error!\n");
			continue;
		}
		printf("Enter the max of y(5-10):");
		cin >> y_1;
		error();
		if (!(y_1 >= 5 && y_1 <= 10))
		{
			printf("Error!\n");
			continue;
		}
		printf("Enter the target(5-15):");
		cin >> max;
		error();
		if (!(max >= 5 && max <= 15))
		{
			printf("Error!\n");
			continue;
		}
		break;
	}
	system("cls");
	windowsize_change(x_1, y_1);
	background_lines(x_1, y_1);
	initialize_1(i, x_1, y_1);
	print(i, x_1, y_1);
	while (check(i, i_temp_2, x_1, y_1))
	{
		SelectUnit(i, i_1, i_temp, x_1, y_1);
		initialize_2(i, temp_1, find_max(i), x_1, y_1);
		if (max == find_max(i) && value == 0)
		{
			gotoxy(hout, 0, 20 + 4 * (y_1 - 4) - 2);
			setcolor(hout, 0, 7);
			printf("You have got the target.Press Enter to continue...");
			value++;
			getchar();
		}
		print(i, x_1, y_1);
		gotoxy(hout, 0, 0);
		setcolor(hout, 0, 7);
		score += iscore(temp_1);
		printf("Score:%d\tMax:%d", score, find_max(i));
		RebuildTempArray(temp_1);
	}
	gotoxy(hout, 0, 20 + 4 * (y_1 - 4) - 1);
	printf("Game Over!\n");
	getchar();
	return 0;
}
