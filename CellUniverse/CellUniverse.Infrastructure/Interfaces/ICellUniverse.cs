using System.Collections.Generic;
using System.Threading.Tasks;
using System.Windows.Media;

namespace CellUniverse.Infrastructure.Interfaces {

    public interface ICellUniverse {

        IEnumerable<Color[,]> GetNext();
    }
}