//1552192 数强 管硕 
#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <io.h>
#include <conio.h>
#include <iomanip>
#include <windows.h>
#include <time.h>
#define MID 100
#define HEIGHT 5
#define SIZE 8
using namespace std;
HANDLE hout = GetStdHandle(STD_OUTPUT_HANDLE);
const char *to_search = ".\\关卡\\Sokoban-Level-???.txt";
char index[17][17] = { 0 };
char index_1[17][17] = { 0 };
char index_2[17][17] = { 0 };
char fileout[10000] = { 0 };
char stepoutfile[100] = { 0 };
int I = 0;
bool FILE_ERR = false;
bool MainMenu = false;
bool CanNotMove = false;
int step = 0;
int Exstep = 0;
int hang, lie;
HWND hwnd;
char ShowStep[1000] = { 0 };
typedef struct retry_map
{
	char c[17][17];
	struct retry_map *pNext;
}MAP;
MAP *pHead = NULL;
void Create()
{
	MAP *pNew;
	pNew = (MAP*)malloc(sizeof(MAP));
	for (int i = 0; i<17; i++)
		for (int j = 0; j<17; j++)
			pNew->c[i][j] = 0;
	for (int i = 0; index[i][0] != 0; i++)
	{
		for (int j = 0; index[i][j] != 0; j++)
		{
			pNew->c[i][j] = index[i][j];
		}
	}
	if (pHead == NULL)
	{
		pNew->pNext = NULL;
		pHead = pNew;
	}
	else
	{
		pNew->pNext = pHead;
		pHead = pNew;
	}
}
void retry()
{
	if (pHead == NULL) return;
	MAP *pTemp = pHead->pNext;
	for (int i = 0; i<17; i++)
		for (int j = 0; j<17; j++)
			index[i][j] = pHead->c[i][j];
	pHead->pNext = NULL;
	free(pHead);
	pHead = pTemp;
}
void windowsize_change(int x = 9, int y = 9)
{
	char change[100];
	sprintf(change, "mode con cols=%d lines=%d", 34 + SIZE*(x - 4), 25 + 4 * (y - 4));
	system(change);
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
void DrawLines()//╠ ╣ ╦ ╩ ╬╔╗╚╝═║
{
	gotoxy(hout, MID - 2, HEIGHT - 1);
	printf("╔══╗");
	for (int i = 0; i<10; i++)
	{
		gotoxy(hout, MID - 2, HEIGHT + i);
		printf("║    ║");
	}
	gotoxy(hout, MID - 2, HEIGHT + 10);
	printf("╚══╝");
}
void GetFiles()
{
	int i = 0, j = 0;
	int k, l;
	int temp;
	int ctrl = 10;
	int choose = 0;
	int key;
	int s = 0;
	char filename[1000][260] = { 0 };
	char filename_1[1000][260] = { 0 };
	long handle;
	char name[260] = ".\\关卡\\";
	struct _finddata_t fileinfo;
	handle = _findfirst(to_search, &fileinfo);
	if (-1 == handle)
	{
		gotoxy(hout, 20, 20);
		printf("There is not a file named level.\n");
		MessageBox(hwnd, "Error:The File Cannot Open.", "WARNING", MB_OK);
		exit(0);
		return;
	}
	if (strlen(fileinfo.name) == 21)
		for (int j = 0; j<260; j++)
			filename_1[i][j] = fileinfo.name[j];
	while (!_findnext(handle, &fileinfo))
	{
		if (strlen(fileinfo.name) == 21)
		{
			if (filename_1[i][0] != 0) i++;
			for (int j = 0; j<260; j++)
				filename_1[i][j] = fileinfo.name[j];
		}
	}
	gotoxy(hout, MID, HEIGHT - 2);
	for (k = 0; filename_1[k][0] != 0; k++)
	{
		filename[k][0] = filename_1[k][14];
		filename[k][1] = filename_1[k][15];
		filename[k][2] = filename_1[k][16];
	}
	printf("Choose the file:");
	while (1)
	{
		l = 0;
		gotoxy(hout, MID, HEIGHT);
		for (temp = ctrl - 10; temp<ctrl; temp++)
		{
			printf("%s", filename[temp]);
			gotoxy(hout, MID, HEIGHT + (++l));
		}
		gotoxy(hout, MID, HEIGHT + choose);
		setcolor(hout, 7, 1);
		printf("%s", filename[ctrl - 10 + choose]);
		key = _getch();
		if (key == 80) choose++;
		if (key == 72) choose--;
		if (key == 13) break;
		if (choose<0)
		{
			choose = 0;
			if (ctrl == 10)
			{
				setcolor(hout, 0, 7);
				continue;
			}
			else if (ctrl>10) ctrl--;
		}
		else if (choose>9)
		{
			choose = 9;
			if (ctrl == i + 1)
			{
				setcolor(hout, 0, 7);
				continue;
			}
			else if (ctrl<i + 1) ctrl++;
		}
		setcolor(hout, 0, 7);
	}
	setcolor(hout, 0, 7);
	_findclose(handle);
	key = ctrl - 10 + choose;
	strcat(name, filename_1[key]);
	strcpy(stepoutfile, name);
	ifstream fin;
	fin.open(name, ios::in);
	if (!fin.is_open())
	{
		MessageBox(hwnd, "Error:The File Cannot Open.", "WARNING", MB_OK);
		return;
	}
	fin.getline(index[0], 16);
	lie = strlen(index[0]);
	for (int p = 0; (fin.getline(index[p + 1], 16)); p++)
		hang = p + 1;
	for (int p = 0; p<hang; p++)
		_strupr(index[p]);
	for (int p = 0; p<hang; p++)
		for (int q = 0; q<lie; q++)
		{
			if (index[p][q] != 'W'&&index[p][q] != 'B'&&index[p][q] != 'S'&&index[p][q] != 'F'&&index[p][q] != 'R'&&index[p][q] != '0'&&index[p][q] != '9' || s>1)
			{
				FILE_ERR = true;
				return;
			}
			else if (index[p][q] == 'S') s++;
		}
	fin.close();
}
void CreateFiles(char *OutFileName)
{
	ofstream fout;
	fout.open(OutFileName, ios::out);
	if (!fout.is_open())
	{
		MessageBox(hwnd, "Error:The File Cannot Open.", "WARNING", MB_OK);
		return;
	}
	fout << fileout << endl;
	fout.close();
	MessageBox(hwnd, "The File Has Been Saved.", "INFORMATION", MB_OK);
}
void save()
{
	char c1[50] = ".\\关卡\\Gamerecord.txt";
	CreateFiles(c1);
}
void PrintUnit(char i, int X, int Y)
{
	if (i == '9')
	{
		setcolor(hout, 0, 0);
		gotoxy(hout, X - 3, Y - 1);
		printf("╔═╗");
		gotoxy(hout, X - 3, Y);
		printf("║  ║");
		gotoxy(hout, X - 3, Y + 1);
		printf("╚═╝");
	}
	else if (i == '0')
	{
		setcolor(hout, 7, 7);
		gotoxy(hout, X - 3, Y - 1);
		printf("╔═╗");
		gotoxy(hout, X - 3, Y);
		printf("║  ║");
		gotoxy(hout, X - 3, Y + 1);
		printf("╚═╝");
	}
	else if (i == 'W')
	{
		setcolor(hout, 4, 0);
		gotoxy(hout, X - 3, Y - 1);
		printf("╔═╗");
		gotoxy(hout, X - 3, Y);
		printf("║XX║");
		gotoxy(hout, X - 3, Y + 1);
		printf("╚═╝");
	}
	else if (i == 'B')
	{
		setcolor(hout, 3, 5);
		gotoxy(hout, X - 3, Y - 1);
		printf("╔═╗");
		gotoxy(hout, X - 3, Y);
		printf("║★║");
		gotoxy(hout, X - 3, Y + 1);
		printf("╚═╝");
	}
	else if (i == 'R')
	{
		setcolor(hout, 6, 4);
		gotoxy(hout, X - 3, Y - 1);
		printf("╔═╗");
		gotoxy(hout, X - 3, Y);
		printf("║★║");
		gotoxy(hout, X - 3, Y + 1);
		printf("╚═╝");
	}
	else if (i == 'S')
	{
		setcolor(hout, 1, 15);
		gotoxy(hout, X - 3, Y - 1);
		printf("╔═╗");
		gotoxy(hout, X - 3, Y);
		printf("║♀║");
		gotoxy(hout, X - 3, Y + 1);
		printf("╚═╝");
	}
	else if (i == 'F')
	{
		setcolor(hout, 11, 12);
		gotoxy(hout, X - 3, Y - 1);
		printf("╔═╗");
		gotoxy(hout, X - 3, Y);
		printf("║●║");
		gotoxy(hout, X - 3, Y + 1);
		printf("╚═╝");
	}
}
void GameScore(char *OutFileName)
{
	char info[100] = { 0 };
	OutFileName[17 + 7] = '-';
	OutFileName[18 + 7] = 'a';
	OutFileName[19 + 7] = 'n';
	OutFileName[20 + 7] = 's';
	OutFileName[21 + 7] = '1';
	OutFileName[22 + 7] = '.';
	OutFileName[23 + 7] = 't';
	OutFileName[24 + 7] = 'x';
	OutFileName[25 + 7] = 't';
	ofstream fout;
	ifstream fin;
	fin.open(OutFileName, ios::in);
	if (!fin.is_open())
	{
		fout.open(OutFileName, ios::out);
		if (!fout.is_open())
		{
			MessageBox(hwnd, "Error:The File Cannot Open.", "WARNING", MB_OK);
			return;
		}
		fout << step << endl;
		MessageBox(hwnd, "The Data File Has Been Written.", "INFORMATION", MB_OK);
	}
	else
	{
		fin >> Exstep;
		if (step<Exstep)
		{
			MessageBox(hwnd, "You Create The Record.The Data Has Changed.", "INFORMATION", MB_OK);
			fout.open(OutFileName, ios::out);
			fout << step;
		}
		else
		{
			sprintf(info, "The Best:%d.\nYour Step:%d.", Exstep, step);
			MessageBox(hwnd, info, "INFORMATION", MB_OK);
		}
	}
}
void FindS(int *x, int *y)
{
	int i, j;
	for (i = 0; i<17; i++)
		for (j = 0; j<17; j++)
			if (index[i][j] == 'S')
			{
				*x = i;
				*y = j;
				return;
			}
}
void up()
{
	int x, y;
	FindS(&x, &y);
	if (index[x][y - 1] == '9' || index[x][y - 1] == 'W')
	{
		CanNotMove = true;
		return;
	}
	else if (y>1 && ((index[x][y - 1] == 'B' || index[x][y - 1] == 'R') && (index[x][y - 2] == 'W' || index[x][y - 2] == 'B' || index[x][y - 2] == 'R')))
	{
		CanNotMove = true;
		return;
	}
	else if (y>1 && ((index[x][y - 1] == 'B' || index[x][y - 1] == 'R') && !(index[x][y - 2] == 'W' || index[x][y - 2] == 'B' || index[x][y - 2] == 'R')))
	{
		if (index_1[x][y - 2] == '0') index[x][y - 2] = 'B';
		else index[x][y - 2] = 'R';
		index[x][y - 1] = 'S';
		if (index_1[x][y] == '0') index[x][y] = '0';
		else index[x][y] = 'F';
	}
	else if (index[x][y - 1] == '0' || index[x][y - 1] == 'F')
	{
		index[x][y - 1] = 'S';
		if (index_1[x][y] == '0') index[x][y] = '0';
		else index[x][y] = 'F';
	}
}
void down()
{
	int x, y;
	FindS(&x, &y);
	if (index[x][y + 1] == '9' || index[x][y + 1] == 'W')
	{
		CanNotMove = true;
		return;
	}
	else if (y<hang - 1 && ((index[x][y + 1] == 'B' || index[x][y + 1] == 'R') && (index[x][y + 2] == 'W' || index[x][y + 2] == 'B' || index[x][y + 2] == 'R')))
	{
		CanNotMove = true;
		return;
	}
	else if (y<hang - 1 && ((index[x][y + 1] == 'B' || index[x][y + 1] == 'R') && !(index[x][y + 2] == 'W' || index[x][y + 2] == 'B' || index[x][y + 2] == 'R')))
	{
		if (index_1[x][y + 2] == '0') index[x][y + 2] = 'B';
		else index[x][y + 2] = 'R';
		index[x][y + 1] = 'S';
		if (index_1[x][y] == '0') index[x][y] = '0';
		else index[x][y] = 'F';
	}
	else if (index[x][y + 1] == '0' || index[x][y + 1] == 'F')
	{
		index[x][y + 1] = 'S';
		if (index_1[x][y] == '0') index[x][y] = '0';
		else index[x][y] = 'F';
	}
}
void left()
{
	int x, y;
	FindS(&x, &y);
	if (index[x - 1][y] == '9' || index[x - 1][y] == 'W')
	{
		CanNotMove = true;
		return;
	}
	else if (x>1 && ((index[x - 1][y] == 'B' || index[x - 1][y] == 'R') && (index[x - 2][y] == 'W' || index[x - 2][y] == 'B' || index[x - 2][y] == 'R')))
	{
		CanNotMove = true;
		return;
	}
	else if (x>1 && ((index[x - 1][y] == 'B' || index[x - 1][y] == 'R') && !(index[x - 2][y] == 'W' || index[x - 2][y] == 'B' || index[x - 2][y] == 'R')))
	{
		if (index_1[x - 2][y] == '0') index[x - 2][y] = 'B';
		else index[x - 2][y] = 'R';
		index[x - 1][y] = 'S';
		if (index_1[x][y] == '0') index[x][y] = '0';
		else index[x][y] = 'F';
	}
	else if (index[x - 1][y] == '0' || index[x - 1][y] == 'F')
	{
		index[x - 1][y] = 'S';
		if (index_1[x][y] == '0') index[x][y] = '0';
		else index[x][y] = 'F';
	}
}
void right()
{
	int x, y;
	FindS(&x, &y);
	if (index[x + 1][y] == '9' || index[x + 1][y] == 'W')
	{
		CanNotMove = true;
		return;
	}
	else if (x<lie - 1 && ((index[x + 1][y] == 'B' || index[x + 1][y] == 'R') && (index[x + 2][y] == 'W' || index[x + 2][y] == 'B' || index[x + 2][y] == 'R')))
	{
		CanNotMove = true;
		return;
	}
	else if (x<lie - 1 && ((index[x + 1][y] == 'B' || index[x + 1][y] == 'R') && !(index[x + 2][y] == 'W' || index[x + 2][y] == 'B' || index[x + 2][y] == 'R')))
	{
		if (index_1[x + 2][y] == '0') index[x + 2][y] = 'B';
		else index[x + 2][y] = 'R';
		index[x + 1][y] = 'S';
		if (index_1[x][y] == '0') index[x][y] = '0';
		else index[x][y] = 'F';
	}
	else if (index[x + 1][y] == '0' || index[x + 1][y] == 'F')
	{
		index[x + 1][y] = 'S';
		if (index_1[x][y] == '0') index[x][y] = '0';
		else index[x][y] = 'F';
	}
}
void reset()
{
	for (int i = 0; index[i][0] != 0; i++)
	{
		for (int j = 0; index[i][j] != 0; j++)
		{
			index[i][j] = index_2[i][j];
		}
	}
}
void control()
{
	int key;
	while (1)
	{
		key = (_getch());
		if (key == 224)
		{
			key = _getch();
			if (key == 80)
			{
				down();
				fileout[I] = 'D';
				I++;
			}
			else if (key == 72)
			{
				up();
				fileout[I] = 'U';
				I++;
			}
			else if (key == 75)
			{
				left();
				fileout[I] = 'L';
				I++;
			}
			else if (key == 77)
			{
				right();
				fileout[I] = 'R';
				I++;
			}
			step++;
			Create();
		}
		else if (key == 113 || key == 81)
		{
			MainMenu = true;
			step = 0;
			return;
		}
		else if (key == 114 || key == 82)
		{
			reset();
		}
		else if (key == 115 || key == 83)
		{
			save();
		}
		else if (key == 8)
		{
			retry();
		}
		else continue;
		break;
	}
}
void Print()
{
	for (int i = 0; index[i][0] != 0; i++)
	{
		for (int j = 0; index[i][j] != 0; j++)
		{
			PrintUnit(index[i][j], 3 + 6 * i, 1 + 3 * j);
		}
	}
	gotoxy(hout, MID, 5 * HEIGHT);
	setcolor(hout, 0, 7);
	printf("STEP:%d", step);
}
void ShowTime(char *ShowStep)
{
	for (int i = 0; ShowStep[i] != 0; i++)
	{
		if (ShowStep[i] == 'L') up();
		else if (ShowStep[i] == 'R') down();
		else if (ShowStep[i] == 'U') left();
		else if (ShowStep[i] == 'D') right();
		Print();
		if (CanNotMove == true)
		{
			MessageBox(hwnd, "Error:The Step Error.", "WARNING", MB_OK);
			CanNotMove = false;
			return;
		}
		Sleep(500);
	}
}
void MapCopy()
{
	for (int i = 0; index[i][0] != 0; i++)
	{
		for (int j = 0; index[i][j] != 0; j++)
		{
			if (index[i][j] == 'B' || index[i][j] == 'S' || index[i][j] == '0') index_1[i][j] = '0';
			else if (index[i][j] == 'R' || index[i][j] == 'F') index_1[i][j] = 'F';
			//else index_1[i][j]='0';
		}
	}
	for (int i = 0; index[i][0] != 0; i++)
	{
		for (int j = 0; index[i][j] != 0; j++)
		{
			index_2[i][j] = index[i][j];
		}
	}
}
void ReadStep(char *FileName)
{
	char ErrInfo[100] = { 0 };
	MapCopy();
	ifstream fin;
	fin.open(FileName, ios::in);
	if (!fin.is_open())
	{
		MessageBox(hwnd, "Error:The File Cannot Open.", "WARNING", MB_OK);
		return;
	}
	for (int i = 0; fin.getline(ShowStep, 1000); i++)
	{
		for (int j = 0; ShowStep[j] != 0; j++)
		{
			if (ShowStep[i] != 'L'&&ShowStep[i] != 'R'&&ShowStep[i] != 'U'&&ShowStep[i] != 'D')
			{
				sprintf(ErrInfo, "Error:The File With The Code %d Error.", i);
				MessageBox(hwnd, ErrInfo, "WARNING", MB_OK);
				return;
			}
		}
		ShowTime(ShowStep);
		for (int k = 0; k<1000; k++) ShowStep[k] = 0;
	}
	MessageBox(hwnd, "DONE.", "INFORMATION", MB_OK);
	fin.close();
}
void GetFiles_1()
{
	int i = 0, j = 0;
	int k, l;
	int temp;
	int ctrl = 10;
	int choose = 0;
	int key;
	int s = 0;
	char filename[1000][260] = { 0 };
	char filename_1[1000][260] = { 0 };
	long handle;
	char name[260] = ".\\关卡\\";
	struct _finddata_t fileinfo;
	handle = _findfirst(to_search, &fileinfo);
	if (-1 == handle)
	{
		gotoxy(hout, 20, 20);
		printf("There is not a file named level.\n");
		MessageBox(hwnd, "Error:The File Cannot Open.", "WARNING", MB_OK);
		exit(0);
		return;
	}
	if (strlen(fileinfo.name) == 21)
		for (int j = 0; j<260; j++)
			filename_1[i][j] = fileinfo.name[j];
	while (!_findnext(handle, &fileinfo))
	{
		if (strlen(fileinfo.name) == 21)
		{
			if (filename_1[i][0] != 0) i++;
			for (int j = 0; j<260; j++)
				filename_1[i][j] = fileinfo.name[j];
		}
	}
	gotoxy(hout, MID, HEIGHT - 2);
	for (k = 0; filename_1[k][0] != 0; k++)
	{
		filename[k][0] = filename_1[k][14];
		filename[k][1] = filename_1[k][15];
		filename[k][2] = filename_1[k][16];
	}
	printf("Choose the file:");
	while (1)
	{
		l = 0;
		gotoxy(hout, MID, HEIGHT);
		for (temp = ctrl - 10; temp<ctrl; temp++)
		{
			printf("%s", filename[temp]);
			gotoxy(hout, MID, HEIGHT + (++l));
		}
		gotoxy(hout, MID, HEIGHT + choose);
		setcolor(hout, 7, 1);
		printf("%s", filename[ctrl - 10 + choose]);
		key = _getch();
		if (key == 80) choose++;
		if (key == 72) choose--;
		if (key == 13) break;
		if (choose<0)
		{
			choose = 0;
			if (ctrl == 10)
			{
				setcolor(hout, 0, 7);
				continue;
			}
			else if (ctrl>10) ctrl--;
		}
		else if (choose>9)
		{
			choose = 9;
			if (ctrl == i + 1)
			{
				setcolor(hout, 0, 7);
				continue;
			}
			else if (ctrl<i + 1) ctrl++;
		}
		setcolor(hout, 0, 7);
	}
	setcolor(hout, 0, 7);
	_findclose(handle);
	key = ctrl - 10 + choose;
	strcat(name, filename_1[key]);
	ifstream fin;
	fin.open(name, ios::in);
	if (!fin.is_open())
	{
		MessageBox(hwnd, "Error:The File Cannot Open.", "WARNING", MB_OK);
		return;
	}
	fin.getline(index[0], 16);
	lie = strlen(index[0]);
	for (int p = 0; (fin.getline(index[p + 1], 16)); p++)
		hang = p + 1;
	for (int p = 0; p<hang; p++)
		_strupr(index[p]);
	for (int p = 0; p<hang; p++)
		for (int q = 0; q<lie; q++)
		{
			if (index[p][q] != 'W'&&index[p][q] != 'B'&&index[p][q] != 'S'&&index[p][q] != 'F'&&index[p][q] != 'R'&&index[p][q] != '0'&&index[p][q] != '9' || s>1)
			{
				FILE_ERR = true;
				return;
			}
			else if (index[p][q] == 'S') s++;
		}
	char ShowStep[10000] = { 0 };
	char *FileName;
	//fin.close();
	FileName = strcat(name, "-ans");
	FileName[17 + 7] = '-';
	FileName[18 + 7] = 'a';
	FileName[19 + 7] = 'n';
	FileName[20 + 7] = 's';
	FileName[21 + 7] = '.';
	FileName[22 + 7] = 't';
	FileName[23 + 7] = 'x';
	FileName[24 + 7] = 't';
	//gotoxy(hout,20,40);
	//printf("0%s\n",FileName);
	fin.close();
	ReadStep(FileName);
}
bool CheckWin()
{
	for (int i = 0; index[i][0] != 0; i++)
	{
		for (int j = 0; index[i][j] != 0; j++)
		{
			if (index[i][j] == 'B') return false;
		}
	}
	return true;
}
void clear()
{
	int i, j;
	for (i = 0; i<17; i++)
		for (j = 0; j<17; j++)
		{
			index[i][j] = 0;
			index_1[i][j] = 0;
			index_2[i][j] = 0;
		}
}
int main(int argc, char **argv)
{
	int ShowOrNot = 0;
	ShowOrNot = MessageBox(hwnd, "Press YES to play,NO to show", "INFORMATION", MB_YESNO);
	windowsize_change(15);
	while (ShowOrNot == IDYES)
	{
		MainMenu = false;
		I = 0;
		DrawLines();
		GetFiles();
		MapCopy();
		if (FILE_ERR)
		{
			gotoxy(hout, 20, 20);
			printf("File Error!\n");
			MessageBox(hwnd, "Error:The File Cannot Open.", "WARNING", MB_OK);
			continue;
		}
		while (!CheckWin())
		{
			Print();
			control();
			if (MainMenu == true) break;
		}
		Print();
		if (CheckWin())
		{
			MessageBox(hwnd, "You Win!", "INFORMATION", MB_OK);
			GameScore(stepoutfile);
		}
		if (MainMenu == true)
		{
			clear();
			system("cls");
			continue;
		}
		system("cls");
		step=0;
	}
	while (ShowOrNot == IDNO)
	{
		MainMenu = false;
		DrawLines();
		GetFiles_1();
		gotoxy(hout, 20, 30);
		//printf("%d\n",strlen(ShowStep));
		//MapCopy();
		if (FILE_ERR)
		{
			gotoxy(hout, 20, 20);
			printf("File Error!\n");
			MessageBox(hwnd, "Error:The File Cannot Open.", "WARNING", MB_OK);
			continue;
		}
	}
	return 0;
}
