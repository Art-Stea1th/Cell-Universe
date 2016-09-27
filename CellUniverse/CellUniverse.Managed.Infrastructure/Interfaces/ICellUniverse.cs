using System.Collections.Generic;
using System.Windows.Media;

namespace CellUniverse.Managed.Infrastructure.Interfaces {

    public interface ICellUniverse {

        IEnumerable<Color[,]> GetNext();
    }
}