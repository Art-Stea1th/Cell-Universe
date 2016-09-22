using System.Windows.Media;


namespace CellUniverse.Infrastructure.Interfaces {

    public interface ICellUniverse {

        Color[,] GetNext();
    }
}