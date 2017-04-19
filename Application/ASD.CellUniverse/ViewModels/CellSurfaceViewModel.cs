using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Prism.Mvvm;

namespace ASD.CellUniverse.ViewModels {

    internal sealed class CellSurfaceViewModel : BindableBase {

        private bool[,] content;

        public bool[,] Content {
            get => content;
            set => SetProperty(ref content, value);
        }
    }
}
