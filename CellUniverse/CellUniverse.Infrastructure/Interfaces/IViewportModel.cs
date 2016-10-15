using System.Collections.Generic;
using System.Windows.Media;

namespace CellUniverse.Infrastructure.Interfaces {

    public interface IViewportModel {

        IEnumerable<Color[,]> NextScreen { get; }
    }
}