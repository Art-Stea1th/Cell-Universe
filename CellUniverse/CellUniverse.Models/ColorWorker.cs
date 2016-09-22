using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Media;

namespace CellUniverse.Models {

    internal sealed class ColorWorker {

        private enum Algorithm { Monochrome, TrueColor }

        private Algorithm algorithm = Algorithm.Monochrome;

        public List<Color> GetColorList(int colorsCount) { // temp impl.

            if (colorsCount != 3) {
                throw new NotImplementedException();
            }

            List<Color> result = new List<Color>(colorsCount);
            result.Add(Color.FromRgb(138, 198, 233));
            result.Add(Color.FromRgb(0, 125, 168));
            result.Add(Color.FromRgb(0, 49, 71));

            return result;
        }
    }
}