#include <QApplication>
#include "configwin.h"

int main(int argc, char *argv[])
{
        QApplication a(argc, argv);
        ConfigWin w;
        w.show();

        return a.exec();
}
