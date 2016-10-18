using System.Collections.Generic;
using System.Windows.Media;

namespace CellUniverse.Infrastructure.Interfaces {

    public interface IUniverseModelColored {

        IEnumerable<Color[,]> NextScreen { get; }
    }
}