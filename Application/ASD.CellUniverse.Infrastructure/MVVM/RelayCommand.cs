using System;
using System.Windows.Input;

namespace ASD.CellUniverse.Infrastructure.MVVM {

    public class RelayCommand : ICommand {

        private readonly Action<object> execute;
        private readonly Predicate<object> canExecute;

        public event EventHandler CanExecuteChanged {
            add => CommandManager.RequerySuggested += value;
            remove => CommandManager.RequerySuggested -= value;
        }

        public RelayCommand(Action<object> execute, Predicate<object> canExecute = null) {
            this.execute = execute ?? throw new ArgumentNullException("execute");
            this.canExecute = canExecute;
        }

        public bool CanExecute(object parameter)
            => canExecute == null ? true : canExecute.Invoke(parameter);

        public void Execute(object parameter) => execute.Invoke(parameter);
    }
}