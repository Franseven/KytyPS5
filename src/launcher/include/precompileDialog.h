#ifndef PRECOMPILE_DIALOG_H
#define PRECOMPILE_DIALOG_H

#include <QByteArray>
#include <QDialog>
#include <QElapsedTimer>
#include <QPair>
#include <QProcess>
#include <QString>
#include <QStringList>
#include <QTimer>
#include <QVector>
#include <cstdint>

class QWidget;
namespace Ui {
class PrecompileDialog;
} // namespace Ui

// Shown when a game is started from the list. Offers to precompile its shaders first.
//
// Nothing is timed and no game runs: `kyty_emulator --precompile` reads every shader the game's
// own files hold, translates it, and stores it in _ShaderCache/<title_id> exactly as one of the
// game's own draws would have. The emulator reports the work on its own stdout — SHADERS, then
// DISTINCT (the number of distinct shaders, which is the denominator), then one PROGRESS line per
// shader, then PRECOMPILED — so the bar is real progress, not an estimate from a clock.
class PrecompileDialog: public QDialog {
	Q_OBJECT

public:
	enum class Mode { Direct, Auto, Manual };

	// interpreter is kyty_emulator.exe, args the argument list already built for a normal run
	// (--game included): the precompilation adds --precompile to it. settings_file is the
	// launcher's settings file, where the choice is remembered per title id.
	//
	// unattended is for a title whose "do not ask again" is set: the choices are hidden and the
	// precompilation starts on its own, leaving only Cancel as a way out. It is still shown,
	// because there is nothing else to watch.
	explicit PrecompileDialog(const QString& game_name, const QString& title_id,
	                          const QString& interpreter, const QStringList& args,
	                          const QString& settings_file, bool unattended = false,
	                          QWidget* parent = nullptr);
	~PrecompileDialog() override;

	[[nodiscard]] Mode GetMode() const { return m_mode; }

	// What the user chose for this title last time. Readable so the caller can honour "do not ask
	// again" without opening the dialog at all.
	struct Prefs {
		Mode     mode    = Mode::Auto;
		uint32_t entries = 0; /* what the last complete precompilation translated */
		bool     ask     = true;
	};

	static Prefs ReadPrefs(const QString& settings_file, const QString& title_id);
	static void  WritePrefs(const QString& settings_file, const QString& title_id,
	                        const Prefs& prefs);

private:
	[[nodiscard]] Mode SelectedMode() const;
	void               StartPrecompile();
	void               StopPrecompile();
	void               ReadOutput();
	void               UpdateEta();

	Ui::PrecompileDialog*            m_ui = nullptr;
	QString                          m_title_id;
	QString                          m_settings_file;
	QString                          m_interpreter;
	QStringList                      m_args;
	Prefs                            m_prefs;
	Mode                             m_mode = Mode::Direct;
	QProcess                         m_process;
	QTimer                           m_eta_timer;
	QElapsedTimer                    m_clock;
	QByteArray                       m_pending; /* stdout bytes not yet forming a whole line */
	uint32_t                         m_total   = 0; /* DISTINCT: what has to be translated */
	uint32_t                         m_done    = 0; /* the last PROGRESS index */
	QVector<QPair<qint64, uint32_t>> m_samples; /* elapsed ms, done, for the rate and the ETA */

	/*slots:*/

	void on_start();
	void on_launch();
	void on_finished(int exit_code, QProcess::ExitStatus status);
	void on_error(QProcess::ProcessError error);

public slots:

	void done(int result) override;
};

#endif // PRECOMPILE_DIALOG_H
