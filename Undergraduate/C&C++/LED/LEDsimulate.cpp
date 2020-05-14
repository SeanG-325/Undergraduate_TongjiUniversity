//1552192	数强	管硕 
#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <conio.h>
#include <iostream>
#include <fstream>
#include <windows.h>
#define SIZE lie
#define LENGTH 33
#define HEIGHT 20
using namespace std;
const char *FileName = "HZK16";
const char *ConfName = "led.conf";
const char *Default = "Welcome!";
typedef BOOL(WINAPI *PROCSETCONSOLEFONT)(HANDLE, DWORD);
HANDLE hout = GetStdHandle(STD_OUTPUT_HANDLE);
HWND hwnd;
int total = 0;
char Data[100][100] = { 0 };
int Code[100][2] = { 0 };
char *Key[100] = { NULL };       //假设最多100字 
char temp[100][32][11] = { 0 };  //假设最多100字 
char temp_1[100][32][11] = { 0 };//假设最多100字 
int hang = 5;
int lie = 10;
int bg = 0;
int fg = 10;
bool PrintWay[8] = { 1,1,1,1,1,1,1,1 };
int ScreenTime = 0;
int LineTime = 0;
char item[20][100] = { 0 };
int Itemcolor[20] = { 0 };
int current = 0;
void ReadConfig()
{
	char temp[200] = { 0 };
	char temp1[200] = { 0 };
	ifstream fin;
	fin.open(ConfName, ios::in);
	if (!fin.is_open())
	{
		MessageBox(hwnd, "The File Cannot Open.", "ERROR", MB_OK);
		return;
	}
	while (!fin.eof())
	{
		char *temp2;
		fin.getline(temp, 200);
		if (strstr(temp, "[setup]") != 0)
		{
			while (!fin.eof())
			{
				fin.getline(temp, 200);
				if (strstr(temp, "[content]") != 0)
				{
					break;
				}
				if (strstr(temp, "行数") != 0)
				{
					hang = temp[5] - '0';
				}
				else if (strstr(temp, "列数") != 0)
				{
					if (temp[5] == '8') lie = 8;
					if (temp[5] == '9') lie = 9;
					if (temp[5] == '1'&&temp[6] == '0') lie = 10;
				}
				else if (strstr(temp, "特效") != 0)
				{
					for (int i = 0; i<6; i++)
					{
						if (temp[4] == i + '1')
						{
							if (temp[6] == 'Y' || temp[6] == 'y')
							{
								if(i>=0&&i<=6)
								PrintWay[i] = true;
							}
							else if (temp[6] == 'N' || temp[6] == 'n')
							{
								if(i>=0&&i<=6)
								PrintWay[i] = false;
							}
							break;
						}
					}
				}
				else if (strstr(temp, "背景色") != 0)
				{
					if (temp[8] == 0)
						for (int i = 0; i<10; i++)
						{
							if (temp[7] == i + '0')
							{
								bg = i;
							}
							if (temp[7] == 'x')
							{
								bg = rand() % 5 + 10;
							}
						}
					if (temp[7] == '1'&&temp[8] != 0)
					{
						for (int i = 0; i<6; i++)
						{
							if (temp[8] == i + '0')
							{
								bg = i + 10;
							}
						}
					}
				}
				else if (strstr(temp, "前景色") != 0)
				{
					if (temp[8] == 0)
						for (int i = 0; i<10; i++)
						{
							if (temp[7] == i + '0')
							{
								fg = i;
							}
							if (temp[7] == 'x')
							{
								do
								{
									fg = rand() % 5 + 10;
								} while (fg != bg);
							}
						}
					if (temp[7] == '1'&&temp[8] != 0)
					{
						for (int i = 0; i<6; i++)
						{
							if (temp[8] == i + '0')
							{
								fg = i + 10;
							}
						}
					}
				}
				else if (strstr(temp, "屏延时") != 0)
				{
					if (temp[7] >= '0'&&temp[7] <= '9')
					{
						ScreenTime = temp[7] - '0';
					}
				}
				else if (strstr(temp, "条延时") != 0)
				{
					if (temp[7] >= '0'&&temp[7] <= '9')
					{
						LineTime = temp[7] - '0';
					}
				}
			}
		}
		if (strstr(temp, "[content]") != 0)
		{
			while (!fin.eof())
			{
				fin.getline(temp, 200);
				if (strstr(temp, "item") != 0 && strstr(temp, "=") != 0 && strstr(temp, "_color") == 0)
				{
					if (!(temp[5] >= '0'&&temp[5] <= '9'))
						for (int i = 0; i<10; i++)
						{
							//system("pause");
							if (temp[4] == i + '1')
							{
								temp2 = strstr(temp, "=");
								temp2++;
								strcpy(temp1, temp2);
								strcpy(item[i], temp1);
							}
						}
					else if (temp[4] == '1'&&temp[5] != 0)
					{
						for (int i = 0; i<6; i++)
						{
							if (temp[5] == i + '1')
							{
								temp2 = strstr(temp, "=");
								temp2++;
								strcpy(temp1, temp2);
								strcpy(item[i + 10], temp1);
							}
						}
					}
					else if (temp[4] == '2'&&temp[5] == '0')
					{
						strcpy(temp1, strstr(temp, "="));
						for (int j = 0; temp1[j] != 0; j++)
						{
							temp1[j] = temp1[j + 1];
						}
						strcpy(item[20], temp1);
					}
				}
				else if (strstr(temp, "item") != 0 && strstr(temp, "=") != 0 && strstr(temp, "_color") != 0)
				{
					if (!(temp[5] >= '0'&&temp[5] <= '9'))
						for (int i = 0; i<10; i++)
						{
							if (temp[4] == i + '1')
							{
								if (temp[14 - 1] == 0)
									for (int j = 0; j<10; j++)
									{
										if (temp[12] == j + '0')
										{
											Itemcolor[i] = j;
										}
									}
								else if (temp[12] == '1'&&temp[12 + 1] != 0)
								{
									for (int j = 0; j<6; j++)
									{
										if (temp[12 + 1] == j + '0')
										{
											Itemcolor[i] = j + 10;
										}
									}
								}
							}
						}
					else if (temp[4] == '1' && (temp[5] >= '0'&&temp[5] <= '9'))
					{
						for (int i = 0; i<6; i++)
						{
							if (temp[5] == i + '1')
							{
								if (temp[14] == 0)
									for (int j = 0; j<10; j++)
									{
										if (temp[14 - 1] == j + '0')
										{
											Itemcolor[i] = j;
										}
									}
								else if (temp[14 - 1] == '1'&&temp[14] != 0)
								{
									for (int j = 0; j<6; j++)
									{
										if (temp[14] == j + '0')
										{
											Itemcolor[i] = j + 10;
										}
									}
								}
							}
						}
					}
					else if (temp[4] == '2'&&temp[5] == '0')
					{
						if (temp[14] == 0)
							for (int j = 0; j<10; j++)
							{
								if (temp[14 - 1] == j + '0')
								{
									Itemcolor[20] = j;
								}
							}
						if (temp[14 - 1] == '1'&&temp[14] != 0)
						{
							for (int j = 0; j<6; j++)
							{
								if (temp[14] == j + '0')
								{
									Itemcolor[20] = j + 10;
								}
							}
						}
					}
				}
			}
		}
	}
	fin.close();
}
void windowsize_change(int x = 5, int y = 10)
{
	char change[100];
	sprintf(change, "mode con cols=%d lines=%d", LENGTH*(y)+2, HEIGHT * (x));
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
void setconsolefont(const HANDLE hout, const int font_no)
{
	HMODULE hKernel32 = GetModuleHandleA("kernel32");
	PROCSETCONSOLEFONT setConsoleFont = (PROCSETCONSOLEFONT)GetProcAddress(hKernel32, "SetConsoleFont");
	/* font_no width high
	0       3     5
	1       4     6
	2       5     8
	3       6     8
	4       8     8
	5       16    8
	6       5     12
	7       6     12
	8       7     12
	9       8     12
	10      16    12
	11      8     16
	12      12    16
	13      8     18
	14      10    18
	15      10    20 */
	setConsoleFont(hout, font_no);
	return;
}
void CodeFormat(char **c)
{
	for (int i = 0; i<100; i++)
	{
		for (int j = 0; j<32; j++)
		{
			_itoa((unsigned char)(c[i][j]), temp[i][j], 2);
			sprintf(temp_1[i][j], "%010s", temp[i][j]);
		}
	}
}
void Print(char **c)//出现式 
{
	int sign = 0;
	for (int i = 0; i<total&&i<lie*hang; i++)
	{
		if (temp[i][0][0] == 0) break;
		//gotoxy(hout,i*LENGTH,0);
		for (int j = 0; j<32; j++)
		{
			gotoxy(hout, i%lie*LENGTH + j % 2 * (LENGTH / 2), j / 2 + i / lie*HEIGHT);
			if (item[0][0] == 0) setcolor(hout, bg, rand() % 5 + 10);
			else setcolor(hout, bg, Itemcolor[current] == 0 ? fg : Itemcolor[current]);
			for (int k = 2; k<10; k++)
			{
				if (temp_1[i + sign*hang*lie][j][k] == '1')
				{
					printf("*");
					printf(" ");
				}
				else if (temp_1[i + sign*hang*lie][j][k] == '0')
				{
					printf(" ");
					printf(" ");
				}
				//Sleep(100);
			}
		}
		if (i + 1 == hang*lie&&total>hang*lie)
		{
			i = -1;
			total -= hang*lie;
			sign++;
			Sleep(ScreenTime * 1000);
			system("cls");
			continue;
		}
	}
}
void Print_1(char **c)//上下扩散式 
{
	int sign = 0;
	for (int i = 0; i<total; i++)
	{
		if (temp[i][0][0] == 0) break;
		//gotoxy(hout,i*LENGTH,0);
		for (int j = 0; j<32; j++)
		{
			gotoxy(hout, i%lie*LENGTH + j % 2 * (LENGTH / 2), j / 2 + i / lie*HEIGHT);
			setcolor(hout, bg, Itemcolor[current] == 0 ? fg : Itemcolor[current]);
			for (int k = 2; k<10; k++)
			{
				if (temp_1[i + sign*hang*lie][j][k] == '1')
				{
					printf("*");
					printf(" ");
				}
				else if (temp_1[i + sign*hang*lie][j][k] == '0')
				{
					printf(" ");
					printf(" ");
				}
			}
			Sleep(10);
		}
		if (i + 1 == hang*lie&&total>hang*lie)
		{
			i = -1;
			total -= hang*lie;
			sign++;
			Sleep(ScreenTime * 1000);
			system("cls");
			continue;
		}
	}
}
void Print_2(char **c)//两边扩散式 
{
	int sign = 0;
	for (int i = 0; i<total; i++)
	{
		if (temp[i][0][0] == 0) break;
		//gotoxy(hout,i*LENGTH,0);
		for (int j = 2; j<10; j++)
		{
			for (int k = 0; k<32; k++)
			{
				gotoxy(hout, i%lie*LENGTH + (k) % 2 * (LENGTH / 2) + 2 * j, k / 2 + i / lie*HEIGHT);
				setcolor(hout, bg, Itemcolor[current] == 0 ? fg : Itemcolor[current]);
				if (temp_1[i + sign*hang*lie][k][j] == '1')
				{
					printf("*");
					printf(" ");
				}
				else if (temp_1[i + sign*hang*lie][k][j] == '0')
				{
					printf(" ");
					printf(" ");
				}
				//Sleep(50);
			}
			Sleep(50);
		}
		if (i + 1 == hang*lie&&total>hang*lie)
		{
			i = -1;
			total -= hang*lie;
			sign++;
			Sleep(ScreenTime * 1000);
			system("cls");
			continue;
		}
	}
}
void Print_3(char **c)//炫彩式 
{
	int sign = 0;
	for (int i = 0; i<total; i++)
	{
		if (temp[i][0][0] == 0) break;
		for (int j = 2; j<10; j++)
		{
			for (int k = 0; k<32; k++)
			{
				gotoxy(hout, i%lie*LENGTH + k % 2 * (LENGTH / 2) + 2 * j, k / 2 + i / lie*HEIGHT);
				setcolor(hout, 0, rand() % 5 + 10);
				if (temp_1[i + sign*hang*lie][k][j] == '1')
				{
					printf("*");
					printf(" ");
				}
				else if (temp_1[i + sign*hang*lie][k][j] == '0')
				{
					printf(" ");
					printf(" ");
				}
			}
			Sleep(50);
		}
		if (i + 1 == hang*lie&&total>hang*lie)
		{
			i = -1;
			total -= hang*lie;
			sign++;
			Sleep(ScreenTime * 1000);
			system("cls");
			continue;
		}
	}
}
void Print_4(char **c)//彩色字式 
{
	int sign = 0;
	for (int i = 0; i<total; i++)
	{
		setcolor(hout, 0, rand() % 5 + 10);
		if (temp[i][0][0] == 0) break;
		for (int j = 2; j<10; j++)
		{
			for (int k = 0; k<32; k++)
			{
				gotoxy(hout, i%lie*LENGTH + k % 2 * (LENGTH / 2) + 2 * j, k / 2 + i / lie*HEIGHT);
				if (temp_1[i+ sign*hang*lie][k][j] == '1')
				{
					printf("*");
					printf(" ");
				}
				else if (temp_1[i+ sign*hang*lie][k][j] == '0')
				{
					printf(" ");
					printf(" ");
				}
			}
			Sleep(50);
		}
		if (i + 1 == hang*lie&&total>hang*lie)
		{
			i = -1;
			total -= hang*lie;
			sign++;
			Sleep(ScreenTime * 1000);
			system("cls");
			continue;
		}
	}
}
void Print_5(char **c)//跳跃横幅式 
{
	int max = 8 * LENGTH;
	for (int ctrl = 0; ctrl<total - SIZE + 1; ctrl++)
	{
		for (int i = 0; i<SIZE; i++)
		{
			if (::temp[i + ctrl][0][0] == 0) break;
			//gotoxy(hout,i*LENGTH,0);
			for (int j = 0; j<32; j++)
			{
				gotoxy(hout, i%lie*LENGTH + j % 2 * (LENGTH / 2), j / 2);
				if (item[0][0] == 0) setcolor(hout, bg, rand() % 5 + 10);
				else setcolor(hout, bg, Itemcolor[current] == 0 ? fg : Itemcolor[current]);
				for (int k = 2; k<10; k++)
				{
					if (temp_1[i + ctrl][j][k] == '1')
					{
						printf("*");
						printf(" ");
					}
					else if (temp_1[i + ctrl][j][k] == '0')
					{
						printf(" ");
						printf(" ");
					}
				}
			}
		}
		Sleep(200);
	}
}
void Print_6(char **c)//一边扩散式  
{
	int sign = 0;
	for (int i = 0; i<total; i++)
	{
		if (temp[i][0][0] == 0) break;
		//gotoxy(hout,i*LENGTH,0);
		for (int j = 2; j<10; j++)
		{
			for (int k = 0; k<32; k += 2)
			{
				gotoxy(hout, i%lie*LENGTH + (k) % 2 * (LENGTH / 2) + 2 * j, k / 2 + i / lie*HEIGHT);//+(k)%2*(LENGTH/2)
				setcolor(hout, bg, Itemcolor[current] == 0 ? fg : Itemcolor[current]);
				if (temp_1[i + sign*hang*lie][k][j] == '1')
				{
					printf("*");
					printf(" ");
				}
				else if (temp_1[i + sign*hang*lie][k][j] == '0')
				{
					printf(" ");
					printf(" ");
				}
				//Sleep(50);
			}
			Sleep(50);
		}
		for (int j = 2; j<10; j++)
		{
			for (int k = 1; k<32; k += 2)
			{
				gotoxy(hout, i%lie*LENGTH + (k) % 2 * (LENGTH / 2) + 2 * j, k / 2 + i / lie*HEIGHT);//+(k)%2*(LENGTH/2)
				setcolor(hout, bg, Itemcolor[current] == 0 ? fg : Itemcolor[current]);
				if (temp_1[i + sign*hang*lie][k][j] == '1')
				{
					printf("*");
					printf(" ");
				}
				else if (temp_1[i + sign*hang*lie][k][j] == '0')
				{
					printf(" ");
					printf(" ");
				}
				//Sleep(50);
			}
			Sleep(50);
		}
		if (i + 1 == hang*lie&&total>hang*lie)
		{
			i = -1;
			total -= hang*lie;
			sign++;
			Sleep(ScreenTime * 1000);
			system("cls");
			continue;
		}
	}
}
void Print_7(char **c)//上下扩散式 
{
	int sign = 0;
	for (int i = 0; i<total; i++)
	{
		if (temp[i][0][0] == 0) break;
		//gotoxy(hout,i*LENGTH,0);
		for (int j = 0; j<17; j++)
		{
			gotoxy(hout, i%lie*LENGTH + j % 2 * (LENGTH / 2), 8-(j+1) / 2 + i / lie*HEIGHT+1);
			setcolor(hout, bg, Itemcolor[current] == 0 ? fg : Itemcolor[current]);
			for (int k = 2; k<10; k++)
			{
				if (temp_1[i + sign*hang*lie][16-j][k] == '1')
				{
					printf("*");
					printf(" ");
				}
				else if (temp_1[i + sign*hang*lie][16-j][k] == '0')
				{
					printf(" ");
					printf(" ");
				}
			}
			gotoxy(hout, i%lie*LENGTH + j % 2 * (LENGTH / 2), 8+j / 2 + i / lie*HEIGHT+1);
			setcolor(hout, bg, Itemcolor[current] == 0 ? fg : Itemcolor[current]);
			for (int k = 2; k<10; k++)
			{
				if (temp_1[i + sign*hang*lie][16+j][k] == '1')
				{
					printf("*");
					printf(" ");
				}
				else if (temp_1[i + sign*hang*lie][16+j][k] == '0')
				{
					printf(" ");
					printf(" ");
				}
			}
			Sleep(50);
		}
		if (i + 1 == hang*lie&&total>hang*lie)
		{
			i = -1;
			total -= hang*lie;
			sign++;
			Sleep(ScreenTime * 1000);
			system("cls");
			continue;
		}
	}
}
void initialize()
{
	for (int i = 0; i<100; i++)
	{
		for (int j = 0; j<33 - 1; j++)
		{
			Key[i][j] = 0;
		}
	}
}
void OpenDataFile()
{
	ifstream fin;
	int way = 0;
	for (int i = 0; i<100; i++)
	{
		Key[i] = new char[32];
	}
	initialize();
	fin.open(FileName, ios::in | ios::binary);
	if (!fin.is_open())
	{
		printf("Error!\n");
		return;
	}
	for (int i = 0; i<100; i++)
	{
		if (Code[i][0] == 0) break;
		fin.seekg((Code[i][0] - 0x00b0) * 3008 + (Code[i][1] - 0x00a1) * 32 + 45120, ios::beg);
		fin.read(Key[i], 32);
	}
	CodeFormat(Key);

	fin.close();
	if (item[0][0] == 0)
	{
		Print(Key);
	}
	else
		while (1)
		{
			way = rand() % 8;
			if (way == 0 && PrintWay[way] != 0)
			{
				Print(Key);
				break;
			}
			else if (way == 1 && PrintWay[way] != 0)
			{
				Print_1(Key);
				break;
			}
			else if (way == 2 && PrintWay[way] != 0)
			{
				Print_2(Key);
				break;
			}
			else if (way == 3 && PrintWay[way] != 0)
			{
				Print_3(Key);
				break;
			}
			else if (way == 4 && PrintWay[way] != 0)
			{
				Print_4(Key);
				break;
			}
			else if (way == 5 && PrintWay[way] != 0)
			{
				Print_5(Key);
				break;
			}
			else if (way == 6 && PrintWay[way] != 0)
			{
				Print_6(Key);
				break;
			}
			else if (way == 7 && PrintWay[way] != 0)
			{
				Print_7(Key);
				break;
			}
		}
}
void DivideCode(char *c)
{
	char *temp = c;
	int max = 0;
	int l = strlen(c);
	int m = 0;
	for (int i = 0; i<l; i++, m++)
	{
		if (*(temp + i) == 0)
		{
			break;
		}
		if ((*(temp + i))>0)
		{
			Code[m][0] = 0x00a3;
			Code[m][1] = (unsigned char)(*(temp + i)) - 32 + 0x00a1 - 1;
		}
		else
		{
			Code[m][0] = (unsigned char)(*(temp + i));
			Code[m][1] = (unsigned char)(*(temp + i + 1));
			i++;
		}
	}
	total = m;
}
void Destruct()
{
	for (int i = 0; i<100; i++)
	{
		if (Key[i] != NULL)
		{
			delete Key[i];
			Key[i] = NULL;
		}
	}
}
void reset()
{
	for (int i = 0; i<100; i++)
	{
		for (int j = 0; j<32; j++)
		{
			for (int k = 0; k<11; k++)
			{
				temp[i][j][k] = 0;
				temp_1[i][j][k] = 0;
			}
		}
	}
	total = 0;
	Destruct();
}

int main()
{
	srand((unsigned int)time(0));
	ReadConfig();
	windowsize_change(hang, lie);
	setconsolefont(hout, 0);
	if (item[0][0] == 0)
	{
		DivideCode((char*)Default);
		OpenDataFile();
		_getch();
	}
	else
		for (int i = 0; i<20 && item[i][0] != 0; i++)
		{
			current = i;
			DivideCode(item[i]);
			//setcolor(hout,bg,fg);
			OpenDataFile();
			reset();
			Sleep(LineTime * 1000);
			system("cls");
		}
	reset();
}
